import torch

from lightx2v_platform.base.global_var import AI_DEVICE


class WanScheduler4ChangingResolutionInterface:
    """把 changing_resolution 能力动态混入任意 WAN scheduler。

    这里不直接继承某个固定的 scheduler，而是在运行时构造一个新类：
    1. 保留 father_scheduler 的基础采样能力；
    2. 用 WanScheduler4ChangingResolution 覆盖 prepare_latents / step_post；
    3. 其余方法仍复用 father_scheduler 的实现。

    这样同一套 changing_resolution 逻辑就能挂到普通 WAN scheduler、
    feature caching scheduler 等不同父类上，而不用为每个父类单独复制代码。
    """

    def __new__(cls, father_scheduler, config):
        class NewClass(WanScheduler4ChangingResolution, father_scheduler):
            def __init__(self, config):
                # 先初始化父 scheduler，再初始化 changing_resolution 相关配置。
                father_scheduler.__init__(self, config)
                WanScheduler4ChangingResolution.__init__(self, config)

        return NewClass(config)


class WanScheduler4ChangingResolution:
    """WAN 的分阶段变分辨率采样逻辑。

    核心思路：
    1. 预先为每个分辨率阶段采样一份噪声 latent；
    2. 正常步沿用父类 step_post 的去噪更新；
    3. 命中切换步时，把当前 x_t 还原成近似 x0，再插值到下一阶段分辨率；
    4. 用下一阶段预先采样好的噪声重新加噪，继续 diffusion 采样。

    注意：
    - resolution_rate 的顺序就是实际分辨率路径；
    - 代码会额外补一份“原始分辨率”的 latent 作为最后阶段；
    - changing_resolution_steps 是 1-based，表示“第 N 步结束后切换分辨率”。
    """

    def __init__(self, config):
        # 默认策略：先在 0.75 倍分辨率上采样，再切回原始分辨率。
        if "resolution_rate" not in config:
            config["resolution_rate"] = [0.75]
        if "changing_resolution_steps" not in config:
            config["changing_resolution_steps"] = [config.infer_steps // 2]

        # 每个切换步都必须对应一个阶段分辨率。
        assert len(config["resolution_rate"]) == len(config["changing_resolution_steps"])

    def prepare_latents(self, seed, latent_shape, dtype=torch.float32):
        """为所有分辨率阶段预先生成噪声 latent。

        参数 latent_shape 是原始目标分辨率的 latent shape，格式为 [C, T, H, W]。
        这里会先按 resolution_rate 依次生成若干阶段 latent，然后额外追加一份原始分辨率 latent。

        例如：
        - resolution_rate = [0.75] 时，真实阶段路径是 [0.75, 1.0]
        - resolution_rate = [1.0, 0.75] 时，真实阶段路径是 [1.0, 0.75, 1.0]
        """

        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(seed)
        self.latents_list = []

        for i in range(len(self.config["resolution_rate"])):
            self.latents_list.append(
                torch.randn(
                    latent_shape[0],
                    latent_shape[1],
                    # H/W 强制取偶数，避免后续 patch / VAE 相关 shape 对齐问题。
                    int(latent_shape[2] * self.config["resolution_rate"][i]) // 2 * 2,
                    int(latent_shape[3] * self.config["resolution_rate"][i]) // 2 * 2,
                    dtype=dtype,
                    device=AI_DEVICE,
                    generator=self.generator,
                )
            )

        # 最后一阶段永远补一份原始分辨率 latent，保证最终能回到目标分辨率。
        self.latents_list.append(
            torch.randn(
                latent_shape[0],
                latent_shape[1],
                latent_shape[2],
                latent_shape[3],
                dtype=dtype,
                device=AI_DEVICE,
                generator=self.generator,
            )
        )

        # 从第 0 个阶段开始采样。
        self.latents = self.latents_list[0]
        # changing_resolution_index 表示当前正在使用第几个分辨率阶段。
        self.changing_resolution_index = 0

    def step_post(self):
        """采样后处理。

        - 非切换步：直接走父类的正常去噪更新；
        - 切换步：改走 step_post_upsample，把 latent 迁移到下一阶段分辨率。

        这里用的是 step_index + 1，所以配置里的 changing_resolution_steps 是 1-based。
        """

        if self.step_index + 1 in self.config["changing_resolution_steps"]:
            self.step_post_upsample()
            # 切换完成后推进阶段索引，供下一步采样以及条件输入读取。
            self.changing_resolution_index += 1
        else:
            super().step_post()

    def step_post_upsample(self):
        """在切换步把当前 latent 迁移到下一阶段分辨率。

        这里不是直接对 noisy latent 做插值，而是：
        1. 先根据当前 x_t 和模型预测噪声 eps 还原近似的 x0；
        2. 对近似 x0 做插值，得到下一分辨率下的“干净 latent”；
        3. 再用下一阶段预采样的噪声重新加噪，回到可继续采样的 x_t。

        这样切分辨率更合理，因为尽量在“内容空间”而不是“噪声空间”迁移。
        """

        # 1. 根据当前采样状态 x_t 和预测噪声 eps，恢复近似的 x0。
        #    WAN 基础 scheduler 中 convert_model_output 的核心也是这一步。
        model_output = self.noise_pred.to(torch.float32)
        sample = self.latents.to(torch.float32)
        sigma_t = self.sigmas[self.step_index]
        x0_pred = sample - sigma_t * model_output
        denoised_sample = x0_pred.to(sample.dtype)

        # 2. 把近似 x0 插值到下一阶段的 shape。
        #    虽然方法名叫 upsample，但这里本质上只是“改尺寸”；
        #    如果下一阶段更小，也会发生 downsample。
        denoised_sample_5d = denoised_sample.unsqueeze(0)  # (C,T,H,W) -> (1,C,T,H,W)

        # latents_list 中每个元素的 shape 都是 [C, T, H, W]；
        # interpolate 的 size 只需要 [T, H, W]，所以取 shape[1:]。
        shape_to_upsampled = self.latents_list[self.changing_resolution_index + 1].shape[1:]
        clean_noise = torch.nn.functional.interpolate(
            denoised_sample_5d,
            size=shape_to_upsampled,
            mode="trilinear",
        )
        clean_noise = clean_noise.squeeze(0)  # (1,C,T,H,W) -> (C,T,H,W)

        # 3. 用“下一阶段预采样的随机噪声”重新加噪，得到下一阶段的 x_t。
        #    好处是分辨率切换后仍然处于 diffusion 轨迹上，而且可被相同 seed 复现。
        noisy_sample = self.add_noise(
            clean_noise,
            self.latents_list[self.changing_resolution_index + 1],
            self.timesteps[self.step_index + 1],
        )

        # 4. 用新的 noisy latent 覆盖当前采样状态。
        self.latents = noisy_sample

        # 如有需要，可在切分辨率后的若干步禁用 corrector，避免跨尺度历史项导致不稳定；
        # 当前实现默认不启用这条策略。
        # self.disable_corrector = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]

        # 5. 重新生成 timesteps / sigmas，并把 shift 调大一点。
        #    父类 set_timesteps 不仅会更新 sigma 日程，还会清空 multistep solver 的历史状态
        #    （如 model_outputs / last_sample / lower_order_nums），因此切尺度也是一次“重置历史”。
        #    提高 shift 的目的，是在新分辨率阶段使用更激进的去噪日程继续采样。
        self.set_timesteps(
            self.infer_steps,
            device=AI_DEVICE,
            shift=self.sample_shift + self.changing_resolution_index + 1,
        )

    def add_noise(self, original_samples, noise, timesteps):
        """把近似 x0 重新加噪成当前阶段可继续采样的 x_t。

        参数说明：
        - original_samples: 插值后的近似 x0
        - noise: 下一阶段预先采样好的随机噪声
        - timesteps: 为了保持接口一致而保留；当前实现未直接使用

        公式：
            x_t = alpha_t * x0 + sigma_t * noise
        """

        # 注意：当前实现使用的是 self.step_index 对应的 sigma，
        # 也就是“以当前切换步的噪声强度”把下一阶段 x0 重新加噪。
        sigma = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples
