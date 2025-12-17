import torch
import base64
import requests
import gc
import json

from lightx2v_platform.base.global_var import AI_DEVICE
torch_device_module = getattr(torch, AI_DEVICE)


class Qwen25_LightllmEncoder:
    def __init__(self, config):
        self.config = config
        self.url = config.get("lightllm_url", "127.0.0.1:8000/generate")
        self.headers = {'Content-Type': 'application/json'}
        
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = config["prompt_template_encode"]
        self.prompt_template_encode_start_idx = config["prompt_template_encode_start_idx"]
        """
        for Qwen-Image-Edit model, CONDITION_IMAGE_SIZE = 1024 * 1024
        for Qwen-Image-Edit-2509 model, CONDITION_IMAGE_SIZE = 384 * 384
        """
        self.CONDITION_IMAGE_SIZE = config.get("CONDITION_IMAGE_SIZE", 384 * 384)
        self.USE_IMAGE_ID_IN_PROMPT = config.get("USE_IMAGE_ID_IN_PROMPT", True)
        self.VAE_IMAGE_SIZE = 1024 * 1024

        self.cpu_offload = config.get("cpu_offload", False)
        self.dtype = torch.bfloat16
        
    def infer_with_lightllm(self, txt, image_list=None):
        data = {
            "inputs": txt,
        }
        if image_list:
            images = []
            for image in image_list:
                image_path = image[1]
                with open(image_path, 'rb') as fin:
                    b64 = base64.b64encode(fin.read()).decode("utf-8")
                images.append({'type': "base64", "data": b64})
            data.update({
                "multimodal_params": {
                    "images": images
                }
            }) 
        
        response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
        resp = response.json()
        return resp["hidden_states"], resp["attention_mask"]
    
    @torch.no_grad()
    def infer(self, text, image_list=None):
        if image_list:
            vae_image_list = []
            vae_image_info_list = []
            if self.USE_IMAGE_ID_IN_PROMPT:
                base_img_prompt = ""
                img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
                for i, image in enumerate(image_list):
                    image = image[0]
                    base_img_prompt += img_prompt_template.format(i + 1)
                    _, vae_image, _, vae_image_info = self.preprocess_image(image)
                    vae_image_list.append(vae_image)
                    vae_image_info_list.append(vae_image_info)
            else:
                base_img_prompt = "<|vision_start|><|image_pad|><|vision_end|>"
                for i, image in enumerate(image_list):
                    image = image[0]
                    _, vae_image, _, vae_image_info = self.preprocess_image(image)
                    vae_image_list.append(vae_image)
                    vae_image_info_list.append(vae_image_info)

            template = self.prompt_template_encode
            drop_idx = self.prompt_template_encode_start_idx
            txt = [template.format(base_img_prompt + e) for e in text]

            image_info = {
                "vae_image_list": vae_image_list,
                "vae_image_info_list": vae_image_info_list,
            }
            hidden_states, attention_mask = self.infer_with_lightllm(txt=txt, image_list=image_list)
            
        else:
            template = self.prompt_template_encode
            drop_idx = self.prompt_template_encode_start_idx
            txt = [template.format(e) for e in text]
            image_info = {}
            hidden_states, attention_mask = self.infer_with_lightllm(txt=txt)
        
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
        encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])

        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=AI_DEVICE)
        prompt_embeds_mask = encoder_attention_mask

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, self.config["num_images_per_prompt"], 1)
        prompt_embeds = prompt_embeds.view(self.config["batchsize"] * self.config["num_images_per_prompt"], seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, self.config["num_images_per_prompt"], 1)
        prompt_embeds_mask = prompt_embeds_mask.view(self.config["batchsize"] * self.config["num_images_per_prompt"], seq_len)

        if self.cpu_offload:
            self.text_encoder.to(torch.device("cpu"))
            torch_device_module.empty_cache()
            gc.collect()

        return prompt_embeds, prompt_embeds_mask, image_info