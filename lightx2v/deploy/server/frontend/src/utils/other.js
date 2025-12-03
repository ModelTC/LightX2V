import { ref, computed, watch, nextTick } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import i18n from './i18n'
import router from '../router'
export const t = i18n.global.t
export const locale = i18n.global.locale

        // 响应式数据
        const loading = ref(false);
        const loginLoading = ref(false);
        const initLoading = ref(false);
        const downloadLoading = ref(false);
        const downloadLoadingMessage = ref('');
        const isLoading = ref(false); // 页面加载loading状态
        const isPageLoading = ref(false); // 分页加载loading状态

        // 录音相关状态
        const isRecording = ref(false);
        const mediaRecorder = ref(null);
        const audioChunks = ref([]);
        const recordingDuration = ref(0);
        const recordingTimer = ref(null);
        const alert = ref({ show: false, message: '', type: 'info' });


        // 短信登录相关数据
        const phoneNumber = ref('');
        const verifyCode = ref('');
        const smsCountdown = ref(0);
        const showSmsForm = ref(false);
        const showErrorDetails = ref(false);
        const showFailureDetails = ref(false);

        // 任务类型下拉菜单
        const showTaskTypeMenu = ref(false);
        const showModelMenu = ref(false);

        // 任务状态轮询相关
        const pollingInterval = ref(null);
        const pollingTasks = ref(new Set()); // 正在轮询的任务ID集合
        const confirmDialog = ref({
            show: false,
            title: '',
            message: '',
            confirmText: '确认', // 使用静态文本，避免翻译依赖
            warning: null,
            confirm: () => { }
        });
        const submitting = ref(false);
        const templateLoading = ref(false); // 模板/任务复用加载状态
        const templateLoadingMessage = ref('');
        const taskSearchQuery = ref('');
        const sidebarCollapsed = ref(false);
        const showExpandHint = ref(false);
        const showGlow = ref(false);
        const isDefaultStateHidden = ref(false);
        const isCreationAreaExpanded = ref(false);
        const hasUploadedContent = ref(false);
        const isContracting = ref(false);
        const faceDetecting = ref(false);  // Face detection loading state
        const audioSeparating = ref(false);  // Audio separation loading state

        const showTaskDetailModal = ref(false);
        const modalTask = ref(null);

        // TTS 模态框状态
        const showVoiceTTSModal = ref(false);
        const showPodcastModal = ref(false);

        // TaskCarousel当前任务状态
        const currentTask = ref(null);

        // 视频加载状态跟踪
        const videoLoadedStates = ref(new Map()); // 跟踪每个视频的加载状态

        // 检查视频是否已加载完成
        const isVideoLoaded = (videoSrc) => {
            return videoLoadedStates.value.get(videoSrc) || false;
        };

        // 设置视频加载状态
        const setVideoLoaded = (videoSrc, loaded) => {
            videoLoadedStates.value.set(videoSrc, loaded);
        };

        // 灵感广场相关数据
        const inspirationSearchQuery = ref('');
        const selectedInspirationCategory = ref('');
        const inspirationItems = ref([]);
        const InspirationCategories = ref([]);

        // 灵感广场分页相关变量
        const inspirationPagination = ref(null);
        const inspirationCurrentPage = ref(1);
        const inspirationPageSize = ref(20);
        const inspirationPageInput = ref(1);
        const inspirationPaginationKey = ref(0);

        // 模板详情弹窗相关数据
        const showTemplateDetailModal = ref(false);
        const selectedTemplate = ref(null);

        // 图片放大弹窗相关数据
        const showImageZoomModal = ref(false);
        const zoomedImageUrl = ref('');

        // 任务文件缓存系统
        const taskFileCache = ref(new Map());
        const taskFileCacheLoaded = ref(false);

        // 模板文件缓存系统
        const templateFileCache = ref(new Map());
        const templateFileCacheLoaded = ref(false);

        // Podcast 音频 URL 缓存系统（模仿任务文件缓存）
        const podcastAudioCache = ref(new Map());
        const podcastAudioCacheLoaded = ref(false);

        // 防重复获取的状态管理
        const templateUrlFetching = ref(new Set()); // 正在获取的URL集合
        const taskUrlFetching = ref(new Map()); // 正在获取的任务URL集合

        // localStorage缓存相关常量
        const TASK_FILE_CACHE_KEY = 'lightx2v_task_files';
        const TEMPLATE_FILE_CACHE_KEY = 'lightx2v_template_files';
        const PODCAST_AUDIO_CACHE_KEY = 'lightx2v_podcast_audio';
        const TASK_FILE_CACHE_EXPIRY = 24 * 60 * 60 * 1000; // 24小时过期
        const PODCAST_AUDIO_CACHE_EXPIRY = 24 * 60 * 60 * 1000; // 24小时过期
        const MODELS_CACHE_KEY = 'lightx2v_models';
        const MODELS_CACHE_EXPIRY = 60 * 60 * 1000; // 1小时过期
        const TEMPLATES_CACHE_KEY = 'lightx2v_templates';
        const TEMPLATES_CACHE_EXPIRY = 24 * 60 * 60 * 1000; // 24小时过期
        const TASKS_CACHE_KEY = 'lightx2v_tasks';
        const TASKS_CACHE_EXPIRY = 5 * 60 * 1000; // 5分钟过期

        const imageTemplates = ref([]);
        const audioTemplates = ref([]);
        const mergedTemplates = ref([]);  // 合并后的模板列表
        const showImageTemplates = ref(false);
        const showAudioTemplates = ref(false);
        const mediaModalTab = ref('history');

        // Template分页相关变量
        const templatePagination = ref(null);
        const templateCurrentPage = ref(1);
        const templatePageSize = ref(20); // 图片模板每页12个，音频模板每页10个
        const templatePageInput = ref(1);
        const templatePaginationKey = ref(0);
        const imageHistory = ref([]);
        const audioHistory = ref([]);
        const ttsHistory = ref([]);

        // 模板文件缓存，避免重复下载
        const currentUser = ref({});
        const models = ref([]);
        const tasks = ref([]);
        const isLoggedIn = ref(null); // null表示未初始化，false表示未登录，true表示已登录

        const selectedTaskId = ref(null);
        const selectedTask = ref(null);
        const selectedModel = ref(null);
        const selectedTaskFiles = ref({ inputs: {}, outputs: {} }); // 存储任务的输入输出文件
        const loadingTaskFiles = ref(false); // 加载任务文件的状态
        const statusFilter = ref('ALL');
        const pagination = ref(null);
        const currentTaskPage = ref(1);
        const taskPageSize = ref(20);
        const taskPageInput = ref(1);
        const paginationKey = ref(0); // 用于强制刷新分页组件
        const taskMenuVisible = ref({}); // 管理每个任务的菜单显示状态
        const nameMap = computed(() => ({
            't2v': t('textToVideo'),
            'i2v': t('imageToVideo'),
            's2v': t('speechToVideo'),
            'animate': t('animate')
        }));

        // 任务类型提示信息
        const taskHints = computed(() => ({
            't2v': [
                t('t2vHint1'),
                t('t2vHint2'),
                t('t2vHint3'),
                t('t2vHint4')
            ],
            'i2v': [
                t('i2vHint1'),
                t('i2vHint2'),
                t('i2vHint3'),
                t('i2vHint4')
            ],
            's2v': [
                t('s2vHint1'),
                t('s2vHint2'),
                t('s2vHint3'),
                t('s2vHint4')
            ],
            'animate': [
                t('animateHint1') || '上传目标角色图片和参考视频',
                t('animateHint2') || '将视频中的角色替换为目标角色',
                ]
        }));

        // 当前任务类型的提示信息
        const currentTaskHints = computed(() => {
            return taskHints.value[selectedTaskId.value] || taskHints.value['s2v'];
        });

        // 滚动提示相关
        const currentHintIndex = ref(0);
        const hintInterval = ref(null);

        // 开始滚动提示
        const startHintRotation = () => {
            if (hintInterval.value) {
                clearInterval(hintInterval.value);
            }
            hintInterval.value = setInterval(() => {
                currentHintIndex.value = (currentHintIndex.value + 1) % currentTaskHints.value.length;
            }, 3000); // 每3秒切换一次
        };

        // 停止滚动提示
        const stopHintRotation = () => {
            if (hintInterval.value) {
                clearInterval(hintInterval.value);
                hintInterval.value = null;
            }
        };

        // 为三个任务类型分别创建独立的表单
        const t2vForm = ref({
            task: 't2v',
            model_cls: '',
            stage: '',
            prompt: '',
            seed: 42
        });

        const i2vForm = ref({
            task: 'i2v',
            model_cls: '',
            stage: '',
            imageFile: null,
            prompt: '',
            seed: 42,
            detectedFaces: []  // List of detected faces: [{ index, bbox, face_image, roleName, ... }]
        });

        const s2vForm = ref({
            task: 's2v',
            model_cls: '',
            stage: '',
            imageFile: null,
            audioFile: null,
            prompt: '',
            seed: 42,
            detectedFaces: [],  // List of detected faces: [{ index, bbox, face_image, roleName, ... }]
            separatedAudios: []  // List of separated audio tracks: [{ speaker_id, audio (base64), roleName, ... }]
        });

        const animateForm = ref({
            task: 'animate',
            model_cls: '',
            stage: '',
            imageFile: null,
            videoFile: null,
            prompt: '视频中的人在做动作',
            seed: 42,
            detectedFaces: []  // List of detected faces: [{ index, bbox, face_image, roleName, ... }]
        });

        // 根据当前选择的任务类型获取对应的表单
        const getCurrentForm = () => {
            switch (selectedTaskId.value) {
                case 't2v':
                    return t2vForm.value;
                case 'i2v':
                    return i2vForm.value;
                case 's2v':
                    return s2vForm.value;
                case 'animate':
                    return animateForm.value;
                default:
                    return t2vForm.value;
            }
        };

        // 控制默认状态显示/隐藏的方法
        const hideDefaultState = () => {
            isDefaultStateHidden.value = true;
        };

        const showDefaultState = () => {
            isDefaultStateHidden.value = false;
        };

        // 控制创作区域展开/收缩的方法
        const expandCreationArea = () => {
            isCreationAreaExpanded.value = true;
            // 添加show类来触发动画
            setTimeout(() => {
                const creationArea = document.querySelector('.creation-area');
                if (creationArea) {
                    creationArea.classList.add('show');
                }
            }, 10);
        };

        const contractCreationArea = () => {
            isContracting.value = true;
            const creationArea = document.querySelector('.creation-area');
            if (creationArea) {
                // 添加hide类来触发收起动画
                creationArea.classList.add('hide');
                creationArea.classList.remove('show');
            }
            // 等待动画完成后更新状态
            setTimeout(() => {
                isCreationAreaExpanded.value = false;
                isContracting.value = false;
                if (creationArea) {
                    creationArea.classList.remove('hide');
                }
            }, 400);
        };

        // 为每个任务类型创建独立的预览变量
        const i2vImagePreview = ref(null);
        const s2vImagePreview = ref(null);
        const s2vAudioPreview = ref(null);
        const animateImagePreview = ref(null);
        const animateVideoPreview = ref(null);

        // 监听上传内容变化
        const updateUploadedContentStatus = () => {
            hasUploadedContent.value = !!(getCurrentImagePreview() || getCurrentAudioPreview() || getCurrentVideoPreview() || getCurrentForm().prompt?.trim());
        };

        // 监听表单变化
        watch([i2vImagePreview, s2vImagePreview, s2vAudioPreview, animateImagePreview, animateVideoPreview, () => getCurrentForm().prompt], () => {
            updateUploadedContentStatus();
        }, { deep: true });

        // 监听任务类型变化，重置提示滚动
        watch(selectedTaskId, () => {
            currentHintIndex.value = 0;
            stopHintRotation();
            startHintRotation();
        });

        // 根据当前任务类型获取对应的预览变量
        const getCurrentImagePreview = () => {
            switch (selectedTaskId.value) {
                case 't2v':
                    return null;
                case 'i2v':
                    return i2vImagePreview.value;
                case 's2v':
                    return s2vImagePreview.value;
                case 'animate':
                    return animateImagePreview.value;
                default:
                    return null;
            }
        };

        const getCurrentAudioPreview = () => {
            switch (selectedTaskId.value) {
                case 't2v':
                    return null
                case 'i2v':
                    return null
                case 's2v':
                    return s2vAudioPreview.value;
                default:
                    return null;
            }
        };

        const setCurrentImagePreview = (value) => {
            switch (selectedTaskId.value) {
                case 't2v':
                    break;
                case 'i2v':
                    i2vImagePreview.value = value;
                    break;
                case 's2v':
                    s2vImagePreview.value = value;
                    break;
                case 'animate':
                    animateImagePreview.value = value;
                    break;
            }
            // 清除图片预览缓存，确保新图片能正确显示
            urlCache.value.delete('current_image_preview');
        };

        const setCurrentAudioPreview = (value) => {
            switch (selectedTaskId.value) {
                case 't2v':
                    break;
                case 'i2v':
                    break;
                case 'animate':
                    break;
                case 's2v':
                    s2vAudioPreview.value = value;
                    break;
            }
            // 清除音频预览缓存，确保新音频能正确显示
            urlCache.value.delete('current_audio_preview');
        };

        // 获取当前任务类型的视频预览
        const getCurrentVideoPreview = () => {
            switch (selectedTaskId.value) {
                case 'animate':
                    return animateVideoPreview.value;
                default:
                    return null;
            }
        };

        // 设置当前任务类型的视频预览
        const setCurrentVideoPreview = (value) => {
            switch (selectedTaskId.value) {
                case 'animate':
                    animateVideoPreview.value = value;
                    break;
            }
            // 清除视频预览缓存，确保新视频能正确显示
            urlCache.value.delete('current_video_preview');
        };

        // 提示词模板相关
        const showTemplates = ref(false);
        const showHistory = ref(false);
        const showPromptModal = ref(false);
        const promptModalTab = ref('templates');

        // 计算属性
        const availableTaskTypes = computed(() => {
            const types = [...new Set(models.value.map(m => m.task))];
            // 重新排序，确保数字人在最左边
            const orderedTypes = [];

            // 检查是否有s2v模型，如果有则添加s2v类型
            const hasS2vModels = models.value.some(m =>
                m.task === 's2v'
            );

            // 优先添加数字人（如果存在相关模型）
            if (hasS2vModels) {
                orderedTypes.push('s2v');
            }

            // 然后添加其他类型
            types.forEach(type => {
                if (type !== 's2v') {
                    orderedTypes.push(type);
                }
            });

            return orderedTypes;
        });

        const availableModelClasses = computed(() => {
            if (!selectedTaskId.value) return [];

            return [...new Set(models.value
                .filter(m => m.task === selectedTaskId.value)
                .map(m => m.model_cls))];
        });

        const filteredTasks = computed(() => {
            let filtered = tasks.value;

            // 状态过滤
            if (statusFilter.value !== 'ALL') {
                filtered = filtered.filter(task => task.status === statusFilter.value);
            }

            // 搜索过滤
            if (taskSearchQuery.value) {
                filtered = filtered.filter(task =>
                task.params.prompt?.toLowerCase().includes(taskSearchQuery.value.toLowerCase()) ||
                task.task_id.toLowerCase().includes(taskSearchQuery.value.toLowerCase()) ||
                    nameMap.value[task.task_type].toLowerCase().includes(taskSearchQuery.value.toLowerCase())
            );
            }

            // 按时间排序，最新的任务在前面
            filtered = filtered.sort((a, b) => {
                const timeA = parseInt(a.create_t) || 0;
                const timeB = parseInt(b.create_t) || 0;
                return timeB - timeA; // 降序排列，最新的在前
            });

            return filtered;
        });

        // 监听状态筛选变化，重置分页到第一页
        watch(statusFilter, (newStatus, oldStatus) => {
            if (newStatus !== oldStatus) {
                currentTaskPage.value = 1;
                taskPageInput.value = 1;
                refreshTasks(true); // 强制刷新
            }
        });

        // 监听搜索查询变化，重置分页到第一页
        watch(taskSearchQuery, (newQuery, oldQuery) => {
            if (newQuery !== oldQuery) {
                currentTaskPage.value = 1;
                taskPageInput.value = 1;
                refreshTasks(true); // 强制刷新
            }
        });

        // 分页信息计算属性，确保响应式更新
        const paginationInfo = computed(() => {
            if (!pagination.value) return null;

            return {
                total: pagination.value.total || 0,
                total_pages: pagination.value.total_pages || 0,
                current_page: pagination.value.current_page || currentTaskPage.value,
                page_size: pagination.value.page_size || taskPageSize.value
            };
        });

        // Template分页信息计算属性
        const templatePaginationInfo = computed(() => {
            if (!templatePagination.value) return null;

            return {
                total: templatePagination.value.total || 0,
                total_pages: templatePagination.value.total_pages || 0,
                current_page: templatePagination.value.current_page || templateCurrentPage.value,
                page_size: templatePagination.value.page_size || templatePageSize.value
            };
        });

        // 灵感广场分页信息计算属性
        const inspirationPaginationInfo = computed(() => {
            if (!inspirationPagination.value) return null;

            return {
                total: inspirationPagination.value.total || 0,
                total_pages: inspirationPagination.value.total_pages || 0,
                current_page: inspirationPagination.value.current_page || inspirationCurrentPage.value,
                page_size: inspirationPagination.value.page_size || inspirationPageSize.value
            };
        });


        // 通用URL缓存
        const urlCache = ref(new Map());

        // 通用URL缓存函数
        const getCachedUrl = (key, urlGenerator) => {
            if (urlCache.value.has(key)) {
                return urlCache.value.get(key);
            }

            const url = urlGenerator();
            urlCache.value.set(key, url);
            return url;
        };

        // 获取历史图片URL（带缓存）
        const getHistoryImageUrl = (history) => {
            if (!history || !history.thumbnail) return '';
            return getCachedUrl(`history_image_${history.filename}`, () => history.thumbnail);
        };

        // 获取用户头像URL（带缓存）
        const getUserAvatarUrl = (user) => {
            if (!user || !user.avatar) return '';
            return getCachedUrl(`user_avatar_${user.username}`, () => user.avatar);
        };

        // 获取当前图片预览URL（带缓存）
        const getCurrentImagePreviewUrl = () => {
            const preview = getCurrentImagePreview();
            if (!preview) return '';
            return getCachedUrl(`current_image_preview`, () => preview);
        };

        // 获取当前音频预览URL（带缓存）
        const getCurrentAudioPreviewUrl = () => {
            const preview = getCurrentAudioPreview();
            if (!preview) return '';
            return getCachedUrl(`current_audio_preview`, () => preview);
        };

        const getCurrentVideoPreviewUrl = () => {
            const preview = getCurrentVideoPreview();
            if (!preview) return '';
            return getCachedUrl(`current_video_preview`, () => preview);
        };

        // Alert定时器，用于清除之前的定时器
        let alertTimeout = null;

        // 方法
        const showAlert = (message, type = 'info', action = null) => {
            // 清除之前的定时器
            if (alertTimeout) {
                clearTimeout(alertTimeout);
                alertTimeout = null;
            }

            // 如果当前有alert正在显示，先关闭它
            if (alert.value && alert.value.show) {
                alert.value.show = false;
                // 等待transition完成（约400ms）后再显示新的alert
                setTimeout(() => {
                    createNewAlert(message, type, action);
                }, 450);
            } else {
                // 如果没有alert在显示，立即创建新的
                // 如果alert存在但已关闭，先重置它以确保状态干净
                if (alert.value && !alert.value.show) {
                    alert.value = { show: false, message: '', type: 'info', action: null };
                }
                // 立即创建新alert，不需要等待nextTick
                createNewAlert(message, type, action);
            }
        };

        // 创建新alert的辅助函数
        const createNewAlert = (message, type, action) => {
            // 再次清除定时器，防止重复设置
            if (alertTimeout) {
                clearTimeout(alertTimeout);
                alertTimeout = null;
            }

            // 创建全新的对象，使用时间戳确保每次都是新对象
            const newAlert = {
                show: true,
                message: String(message),
                type: String(type),
                action: action ? {
                    label: String(action.label),
                    onClick: action.onClick
                } : null,
                // 添加一个时间戳确保每次都是新对象，用于key
                _timestamp: Date.now()
            };

            // 直接赋值新对象
            alert.value = newAlert;

            // 设置自动关闭定时器
            alertTimeout = setTimeout(() => {
                if (alert.value && alert.value.show && alert.value._timestamp === newAlert._timestamp) {
                alert.value.show = false;
                }
                alertTimeout = null;
            }, 5000);
        };

        // 显示确认对话框
        const showConfirmDialog = (options) => {
            return new Promise((resolve) => {
                confirmDialog.value = {
                    show: true,
                    title: options.title || '确认操作',
                    message: options.message || '确定要执行此操作吗？',
                    confirmText: options.confirmText || '确认',
                    warning: options.warning || null,
                    confirm: () => {
                        confirmDialog.value.show = false;
                        resolve(true);
                    },
                    cancel: () => {
                        confirmDialog.value.show = false;
                        resolve(false);
                    }
                };
            });
        };

        const setLoading = (value) => {
            loading.value = value;
        };

        const apiCall = async (endpoint, options = {}) => {
            const url = `${endpoint}`;
            const headers = {
                'Content-Type': 'application/json',
                ...options.headers
            };

            if (localStorage.getItem('accessToken')) {
                headers['Authorization'] = `Bearer ${localStorage.getItem('accessToken')}`;
            }

            const response = await fetch(url, {
                ...options,
                headers
            });

            if (response.status === 401) {
                logout(false);
                showAlert(t('authFailedPleaseRelogin'), 'warning', {
                    label: t('login'),
                    onClick: login
                });
                throw new Error(t('authFailedPleaseRelogin'));
            }
            if (response.status === 400) {
                const error = await response.json();
                showAlert(error.message, 'danger');
                throw new Error(error.message);
            }

            // 添加50ms延迟，防止触发服务端频率限制
            await new Promise(resolve => setTimeout(resolve, 50));

            return response;
        };

        const loginWithGitHub = async () => {
            try {
                console.log('starting GitHub login')
                const response = await fetch('/auth/login/github');
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                localStorage.setItem('loginSource', 'github');
                window.location.href = data.auth_url;
            } catch (error) {
                console.log('GitHub login error:', error);
                showAlert(t('getGitHubAuthUrlFailed'), 'danger');
            }
        };

        const loginWithGoogle = async () => {
            try {
                console.log('starting Google login')
                const response = await fetch('/auth/login/google');
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                localStorage.setItem('loginSource', 'google');
                window.location.href = data.auth_url;
            } catch (error) {
                console.error('Google login error:', error);
                showAlert(t('getGoogleAuthUrlFailed'), 'danger');
            }
        };

        // 发送短信验证码
        const sendSmsCode = async () => {
            if (!phoneNumber.value) {
                showAlert(t('pleaseEnterPhoneNumber'), 'warning');
                return;
            }

            // 简单的手机号格式验证
            const phoneRegex = /^1[3-9]\d{9}$/;
            if (!phoneRegex.test(phoneNumber.value)) {
                showAlert(t('pleaseEnterValidPhoneNumber'), 'warning');
                return;
            }

            try {
                const response = await fetch(`./auth/login/sms?phone_number=${phoneNumber.value}`);
                const data = await response.json();

                if (response.ok) {
                    showAlert(t('verificationCodeSent'), 'success');
                    // 开始倒计时
                    startSmsCountdown();
                } else {
                    showAlert(data.message || t('sendVerificationCodeFailed'), 'danger');
                }
            } catch (error) {
                showAlert(t('sendVerificationCodeFailedRetry'), 'danger');
            }
        };

        // 短信验证码登录
        const loginWithSms = async () => {
            if (!phoneNumber.value || !verifyCode.value) {
                showAlert(t('pleaseEnterPhoneAndCode'), 'warning');
                return;
            }

            try {
                const response = await fetch(`./auth/callback/sms?phone_number=${phoneNumber.value}&verify_code=${verifyCode.value}`);
                const data = await response.json();

                if (response.ok) {
                    localStorage.setItem('accessToken', data.access_token);
                    if (data.refresh_token) {
                        localStorage.setItem('refreshToken', data.refresh_token);
                    }
                    localStorage.setItem('currentUser', JSON.stringify(data.user_info));
                    currentUser.value = data.user_info;

                    // 登录成功后初始化数据
                    await init();

                    router.push('/generate');
                    console.log('login with sms success');
                    isLoggedIn.value = true;

                    showAlert(t('loginSuccess'), 'success');
                } else {
                    showAlert(data.message || t('verificationCodeErrorOrExpired'), 'danger');
                }
            } catch (error) {
                showAlert(t('loginFailedRetry'), 'danger');
            }
        };

        // 处理手机号输入框回车键
        const handlePhoneEnter = () => {
            if (phoneNumber.value && !smsCountdown.value) {
                sendSmsCode();
            }
        };

        // 处理验证码输入框回车键
        const handleVerifyCodeEnter = () => {
            if (phoneNumber.value && verifyCode.value) {
                loginWithSms();
            }
        };

        // 移动端检测和样式应用
        const applyMobileStyles = () => {
            if (window.innerWidth <= 640) {
                // 为左侧功能区添加移动端样式
                const leftNav = document.querySelector('.relative.w-20.pl-5.flex.flex-col.z-10');
                if (leftNav) {
                    leftNav.classList.add('mobile-bottom-nav');
                }

                // 为导航按钮容器添加移动端样式
                const navContainer = document.querySelector('.p-2.flex.flex-col.justify-center.h-full');
                if (navContainer) {
                    navContainer.classList.add('mobile-nav-buttons');
                }

                // 为所有导航按钮添加移动端样式
                const navButtons = document.querySelectorAll('.relative.w-20.pl-5.flex.flex-col.z-10 button');
                navButtons.forEach(btn => {
                    btn.classList.add('mobile-nav-btn');
                });

                // 为主内容区域添加移动端样式
                const contentAreas = document.querySelectorAll('.flex-1.flex.flex-col.min-h-0');
                contentAreas.forEach(area => {
                    area.classList.add('mobile-content');
                });
            }
        };

        // 短信验证码倒计时
        const startSmsCountdown = () => {
            smsCountdown.value = 60;
            const timer = setInterval(() => {
                smsCountdown.value--;
                if (smsCountdown.value <= 0) {
                    clearInterval(timer);
                }
            }, 1000);
        };

        // 切换短信登录表单显示
        const toggleSmsLogin = () => {
            showSmsForm.value = !showSmsForm.value;
            if (!showSmsForm.value) {
                // 重置表单数据
                phoneNumber.value = '';
                verifyCode.value = '';
                smsCountdown.value = 0;
            }
        };

        const handleLoginCallback = async (code, source) => {
            try {
                const response = await fetch(`/auth/callback/${source}?code=${code}`);
                if (response.ok) {
                    const data = await response.json();
                    console.log(data);
                    localStorage.setItem('accessToken', data.access_token);
                    if (data.refresh_token) {
                        localStorage.setItem('refreshToken', data.refresh_token);
                    }
                    localStorage.setItem('currentUser', JSON.stringify(data.user_info));
                    currentUser.value = data.user_info;
                    isLoggedIn.value = true;

                    // 在进入新页面前显示loading
                    isLoading.value = true;

                    // 登录成功后初始化数据
                    await init();

                    // 检查是否有分享数据需要导入
                    const shareData = localStorage.getItem('shareData');
                    if (shareData) {
                        // 解析分享数据获取shareId
                        try {
                            const parsedShareData = JSON.parse(shareData);
                            const shareId = parsedShareData.share_id || parsedShareData.task_id;
                            if (shareId) {
                                localStorage.removeItem('shareData');
                                // 跳转回分享页面，让createSimilar函数处理数据
                                router.push(`/share/${shareId}`);
                                return;
                            }
                        } catch (error) {
                            console.warn('Failed to parse share data:', error);
                        }
                        localStorage.removeItem('shareData');
                    }

                    // 默认跳转到生成页面
                    router.push('/generate');
                    console.log('login with callback success');

                    // 清除URL中的code参数
                    window.history.replaceState({}, document.title, window.location.pathname);
                } else {
                    const error = await response.json();
                    showAlert(`${t('loginFailedRetry')}: ${error.detail}`, 'danger');
                }
            } catch (error) {
                showAlert(t('loginError'), 'danger');
                console.error(error);
            }
        };

        let refreshPromise = null;

        const logout = (showMessage = true) => {
            localStorage.removeItem('accessToken');
            localStorage.removeItem('refreshToken');
            localStorage.removeItem('currentUser');
            refreshPromise = null;

            clearAllCache();
            switchToLoginView();
            isLoggedIn.value = false;

            models.value = [];
            tasks.value = [];
            if (showMessage) {
                showAlert(t('loggedOut'), 'info');
            }
        };

        const login = () => {
            switchToLoginView();
            isLoggedIn.value = false;
        };

        const loadModels = async (forceRefresh = false) => {
            try {
                // 如果不是强制更新，先尝试从缓存加载
                if (!forceRefresh) {
                    const cachedModels = loadFromCache(MODELS_CACHE_KEY, MODELS_CACHE_EXPIRY);
                    if (cachedModels) {
                        console.log('成功从缓存加载模型列表');
                        models.value = cachedModels;
                        return;
                        }
                }

                console.log('开始加载模型列表...');
                const response = await apiRequest('/api/v1/model/list');
                if (response && response.ok) {
                    const data = await response.json();
                    console.log('模型列表数据:', data);
                    const modelsData = data.models || [];
                    models.value = modelsData;

                    // 保存到缓存
                    saveToCache(MODELS_CACHE_KEY, modelsData);
                    console.log('模型列表已缓存');
                } else if (response) {
                    console.error('模型列表API响应失败:', response);
                    showAlert(t('loadModelListFailed'), 'danger');
                }
                // 如果response为null，说明是认证错误，apiRequest已经处理了
            } catch (error) {
                console.error('加载模型失败:', error);
                showAlert(`${t('loadModelFailed')}: ${error.message}`, 'danger');
            }
        };

        const refreshTemplateFileUrl = (templatesData) => {
            for (const img of templatesData.images) {
                console.log('刷新图片素材文件URL:', img.filename, img.url);
                setTemplateFileToCache(img.filename, {url: img.url, timestamp: Date.now()});
            }
            for (const audio of templatesData.audios) {
                console.log('刷新音频素材文件URL:', audio.filename, audio.url);
                setTemplateFileToCache(audio.filename, {url: audio.url, timestamp: Date.now()});
            }
            for (const video of templatesData.videos) {
                console.log('刷新视频素材文件URL:', video.filename, video.url);
                setTemplateFileToCache(video.filename, {url: video.url, timestamp: Date.now()});
            }
        }

        // 加载模板文件
        const loadImageAudioTemplates = async (forceRefresh = false) => {
            try {
                // 如果不是强制刷新，先尝试从缓存加载
                const cacheKey = `${TEMPLATES_CACHE_KEY}_IMAGE_AUDIO_MERGED_${templateCurrentPage.value}_${templatePageSize.value}`;
                if (!forceRefresh) {
                // 构建缓存键，包含分页和过滤条件
                const cachedTemplates = loadFromCache(cacheKey, TEMPLATES_CACHE_EXPIRY);
                    if (cachedTemplates && cachedTemplates.templates) {
                    console.log('成功从缓存加载模板列表');
                        // 优先使用合并后的模板列表
                        if (cachedTemplates.templates.merged) {
                            mergedTemplates.value = cachedTemplates.templates.merged || [];
                            // 从合并列表中提取图片和音频
                            const images = [];
                            const audios = [];
                            mergedTemplates.value.forEach(template => {
                                if (template.image) {
                                    images.push(template.image);
                                }
                                if (template.audio) {
                                    audios.push(template.audio);
                                }
                            });
                            imageTemplates.value = images;
                            audioTemplates.value = audios;
                        } else {
                            // 向后兼容：如果没有合并列表，使用旧的格式
                            imageTemplates.value = cachedTemplates.templates.images || [];
                            audioTemplates.value = cachedTemplates.templates.audios || [];
                        }
                        templatePagination.value = cachedTemplates.pagination || null;
                    return;
                    }
                }

                console.log('开始加载图片音乐素材库...');
                const response = await publicApiCall(`/api/v1/template/list?page=${templateCurrentPage.value}&page_size=${templatePageSize.value}`);
                if (response.ok) {
                    const data = await response.json();
                    console.log('图片音乐素材库数据:', data);

                    // 使用合并后的模板列表
                    const merged = data.templates?.merged || [];
                    mergedTemplates.value = merged;

                    // 为了保持向后兼容，从合并列表中提取图片和音频
                    const images = [];
                    const audios = [];
                    merged.forEach(template => {
                        if (template.image) {
                            images.push(template.image);
                        }
                        if (template.audio) {
                            audios.push(template.audio);
                        }
                    });

                    refreshTemplateFileUrl({ images, audios, videos: data.templates?.videos || [] });
                    const templatesData = {
                        images: images,
                        audios: audios,
                        merged: merged
                    };

                    imageTemplates.value = images;
                    audioTemplates.value = audios;
                    templatePagination.value = data.pagination || null;

                    // 保存到缓存
                    saveToCache(cacheKey, {
                        templates: templatesData,
                        pagination: templatePagination.value
                    });
                    console.log('图片音乐素材库已缓存:', templatesData);

                } else {
                    console.warn('加载素材库失败');
                }
            } catch (error) {
                console.warn('加载素材库失败:', error);
            }
        };

        // 获取素材文件的通用函数（带缓存）
        const getTemplateFile = async (template) => {
            const cacheKey = template.url;

            // 先检查内存缓存
            if (templateFileCache.value.has(cacheKey)) {
                console.log('从内存缓存获取素材文件:', template.filename);
                return templateFileCache.value.get(cacheKey);
            }

            // 如果缓存中没有，则下载并缓存
            console.log('下载素材文件:', template.filename);
            const response = await fetch(template.url, {
                cache: 'force-cache' // 强制使用浏览器缓存
            });

            if (response.ok) {
                const blob = await response.blob();

                // 根据文件扩展名确定正确的MIME类型
                let mimeType = blob.type;
                const extension = template.filename.toLowerCase().split('.').pop();

                if (extension === 'wav') {
                    mimeType = 'audio/wav';
                } else if (extension === 'mp3') {
                    mimeType = 'audio/mpeg';
                } else if (extension === 'm4a') {
                    mimeType = 'audio/mp4';
                } else if (extension === 'ogg') {
                    mimeType = 'audio/ogg';
                } else if (extension === 'webm') {
                    mimeType = 'audio/webm';
                }

                console.log('文件扩展名:', extension, 'MIME类型:', mimeType);

                const file = new File([blob], template.filename, { type: mimeType });

                // 缓存文件对象
                templateFileCache.value.set(cacheKey, file);
                console.log('下载素材文件完成:', template.filename);
                return file;
            } else {
                throw new Error('下载素材文件失败');
            }
        };

        // 选择图片素材
        const selectImageTemplate = async (template) => {
            try {
                const file = await getTemplateFile(template);

                if (selectedTaskId.value === 'i2v') {
                    i2vForm.value.imageFile = file;
                    i2vForm.value.detectedFaces = [];  // Reset detected faces
                } else if (selectedTaskId.value === 's2v') {
                    s2vForm.value.imageFile = file;
                    s2vForm.value.detectedFaces = [];  // Reset detected faces
                } else if (selectedTaskId.value === 'animate') {
                    animateForm.value.imageFile = file;
                    animateForm.value.detectedFaces = [];  // Reset detected faces
                }

                // 获取图片的 http/https URL（用于人脸识别和预览）
                let imageUrl = null;
                // 优先使用 template.url（如果是 http/https URL）
                if (template.url && (template.url.startsWith('http://') || template.url.startsWith('https://'))) {
                    imageUrl = template.url;
                } else if (template.inputs && template.inputs.input_image) {
                    // 如果有 inputs.input_image，使用 getTemplateFileUrlAsync 获取 URL
                    imageUrl = await getTemplateFileUrlAsync(template.inputs.input_image, 'images');
                } else if (template.filename) {
                    // 如果有 filename，尝试使用 getTemplateFileUrlAsync
                    imageUrl = await getTemplateFileUrlAsync(template.filename, 'images');
                }

                // 创建预览（使用 data URL 作为预览）
                const reader = new FileReader();
                reader.onload = async (e) => {
                    const imageDataUrl = e.target.result;
                    // 如果有 http/https URL，使用它作为预览；否则使用 data URL
                    setCurrentImagePreview(imageUrl || imageDataUrl);
                    updateUploadedContentStatus();
                    showImageTemplates.value = false;
                    showAlert(t('imageTemplateSelected'), 'success');
                    // 不再自动检测人脸，等待用户手动打开多角色模式开关
                };
                reader.readAsDataURL(file);

            } catch (error) {
                showAlert(`${t('loadImageTemplateFailed')}: ${error.message}`, 'danger');
            }
        };

        // 选择音频素材
        const selectAudioTemplate = async (template) => {
            try {
                const file = await getTemplateFile(template);

                s2vForm.value.audioFile = file;

                // 创建预览
                const reader = new FileReader();
                reader.onload = (e) => {
                    setCurrentAudioPreview(e.target.result);
                    updateUploadedContentStatus();
                };
                reader.readAsDataURL(file);

                showAudioTemplates.value = false;
                showAlert(t('audioTemplateSelected'), 'success');
            } catch (error) {
                showAlert(`${t('loadAudioTemplateFailed')}: ${error.message}`, 'danger');
            }
        };

        // 预览音频素材
        const previewAudioTemplate = (template) => {
            console.log('预览音频模板:', template);
            const audioUrl = getTemplateFileUrl(template.filename, 'audios');
            console.log('音频URL:', audioUrl);
            if (!audioUrl) {
                showAlert(t('audioFileUrlFailed'), 'danger');
                return;
            }

            // 停止当前播放的音频
            if (currentPlayingAudio) {
                currentPlayingAudio.pause();
                currentPlayingAudio.currentTime = 0;
                currentPlayingAudio = null;
            }

            const audio = new Audio(audioUrl);
            currentPlayingAudio = audio;

            // 监听音频播放结束事件
            audio.addEventListener('ended', () => {
                currentPlayingAudio = null;
                // 调用停止回调
                if (audioStopCallback) {
                    audioStopCallback();
                    audioStopCallback = null;
                }
            });

            audio.addEventListener('error', () => {
                console.error('音频播放失败:', audio.error);
                showAlert(t('audioPlaybackFailed'), 'danger');
                currentPlayingAudio = null;
                // 调用停止回调
                if (audioStopCallback) {
                    audioStopCallback();
                    audioStopCallback = null;
                }
            });

            audio.play().catch(error => {
                console.error('音频播放失败:', error);
                showAlert(t('audioPlaybackFailed'), 'danger');
                currentPlayingAudio = null;
            });
        };

        const handleImageUpload = async (event) => {
            const file = event.target.files[0];
            if (file) {
                if (selectedTaskId.value === 'i2v') {
                    i2vForm.value.imageFile = file;
                    i2vForm.value.detectedFaces = [];  // Reset detected faces
                } else if (selectedTaskId.value === 's2v') {
                    s2vForm.value.imageFile = file;
                    s2vForm.value.detectedFaces = [];  // Reset detected faces
                } else if (selectedTaskId.value === 'animate') {
                    animateForm.value.imageFile = file;
                    animateForm.value.detectedFaces = [];  // Reset detected faces
                }
                const reader = new FileReader();
                reader.onload = async (e) => {
                    const imageDataUrl = e.target.result;
                    setCurrentImagePreview(imageDataUrl);
                    updateUploadedContentStatus();

                    // 不再自动检测人脸，等待用户手动打开多角色模式开关
                };
                reader.readAsDataURL(file);
            } else {
                // 用户取消了选择，保持原有图片不变
                // 不做任何操作
            }
        };

        // Crop face image from original image based on bbox coordinates
        const cropFaceImage = (imageUrl, bbox) => {
            return new Promise((resolve, reject) => {
                // Validate bbox
                if (!bbox || bbox.length !== 4) {
                    reject(new Error('Invalid bbox coordinates'))
                    return
                }

                const [x1, y1, x2, y2] = bbox
                const width = x2 - x1
                const height = y2 - y1

                if (width <= 0 || height <= 0) {
                    reject(new Error(`Invalid bbox dimensions: ${width}x${height}`))
                    return
                }

                const img = new Image()

                // For data URLs, crossOrigin is not needed
                if (imageUrl.startsWith('data:')) {
                    img.onload = () => {
                        try {
                            // Create Canvas to crop image
                            const canvas = document.createElement('canvas')
                            canvas.width = width
                            canvas.height = height
                            const ctx = canvas.getContext('2d')

                            // Draw the cropped region to Canvas
                            ctx.drawImage(
                                img,
                                x1, y1, width, height,  // Source image crop region
                                0, 0, width, height     // Canvas drawing position
                            )

                            // Convert to base64
                            const base64 = canvas.toDataURL('image/png')
                            resolve(base64)
                        } catch (error) {
                            reject(error)
                        }
                    }
                    img.onerror = (e) => {
                        reject(new Error('Failed to load image for cropping'))
                    }
                    img.src = imageUrl
                } else {
                    // For other URLs, set crossOrigin
                    img.crossOrigin = 'anonymous'
                    img.onload = () => {
                        try {
                            // Create Canvas to crop image
                            const canvas = document.createElement('canvas')
                            canvas.width = width
                            canvas.height = height
                            const ctx = canvas.getContext('2d')

                            // Draw the cropped region to Canvas
                            ctx.drawImage(
                                img,
                                x1, y1, width, height,  // Source image crop region
                                0, 0, width, height     // Canvas drawing position
                            )

                            // Convert to base64
                            const base64 = canvas.toDataURL('image/png')
                            resolve(base64)
                        } catch (error) {
                            reject(error)
                        }
                    }
                    img.onerror = (e) => {
                        reject(new Error('Failed to load image for cropping (CORS or network error)'))
                    }
                    img.src = imageUrl
                }
            })
        }

        // Detect faces in uploaded image
        const detectFacesInImage = async (imageDataUrl) => {
            try {
                // 验证输入
                if (!imageDataUrl || imageDataUrl.trim() === '') {
                    console.error('detectFacesInImage: imageDataUrl is empty');
                    return;
                }

                faceDetecting.value = true;

                // Convert blob URL to data URL (backend can't access blob URLs)
                // For http/https URLs, send directly to backend
                let imageInput = imageDataUrl;
                if (imageDataUrl.startsWith('blob:')) {
                    // Blob URL: convert to data URL since backend can't access blob URLs
                    try {
                        const response = await fetch(imageDataUrl);
                        if (!response.ok) {
                            throw new Error(`Failed to fetch image: ${response.statusText}`);
                        }
                        const blob = await response.blob();
                        imageInput = await new Promise((resolve, reject) => {
                            const reader = new FileReader();
                            reader.onload = () => resolve(reader.result);
                            reader.onerror = reject;
                            reader.readAsDataURL(blob);
                        });
                    } catch (error) {
                        console.error('Failed to convert blob URL to data URL:', error);
                        throw error;
                    }
                }
                // For data URLs and http/https URLs, send directly to backend

                // 再次验证 imageInput
                if (!imageInput || imageInput.trim() === '') {
                    console.error('detectFacesInImage: imageInput is empty after processing');
                    return;
                }

                const response = await apiCall('/api/v1/face/detect', {
                    method: 'POST',
                    body: JSON.stringify({
                        image: imageInput
                    })
                });

                if (!response.ok) {
                    console.error('Face detection failed:', response.status, response.statusText);
                    return;
                }

                const data = await response.json();
                console.log('Face detection response:', data);

                if (data && data.faces) {
                    // Crop face images for each detected face
                    // Use the original imageDataUrl for cropping (cropFaceImage can handle both data URLs and regular URLs)
                    const facesWithImages = await Promise.all(
                        data.faces.map(async (face, index) => {
                            try {
                                // Crop face image from original image based on bbox
                                const croppedImage = await cropFaceImage(imageDataUrl, face.bbox)

                                // Remove data URL prefix, keep only base64 part (consistent with backend format)
                                // croppedImage is in format: "data:image/png;base64,xxxxx"
                                let base64Data = croppedImage
                                if (croppedImage.includes(',')) {
                                    base64Data = croppedImage.split(',')[1]
                                }

                                if (!base64Data) {
                                    console.error(`Failed to extract base64 from cropped image for face ${index}`)
                                    base64Data = null
                                }

                                return {
                                    ...face,
                                    face_image: base64Data,  // Base64 encoded face region image (without data URL prefix)
                                    roleName: `角色${index + 1}`,
                                    isEditing: false  // Track editing state for each face
                                }
                            } catch (error) {
                                console.error(`Failed to crop face ${index}:`, error, 'bbox:', face.bbox);
                                // Return face without face_image if cropping fails
                                return {
                                    ...face,
                                    face_image: null,
                                    roleName: `角色${index + 1}`,
                                    isEditing: false
                                }
                            }
                        })
                    );

                    const currentForm = getCurrentForm();
                    if (currentForm) {
                        currentForm.detectedFaces = facesWithImages;
                        console.log('Updated detectedFaces:', currentForm.detectedFaces.length, 'faces with images');
                        // 音频分离由统一的 watch 监听器处理，不需要在这里手动调用
                    }
                }
            } catch (error) {
                console.error('Face detection error:', error);
                // Silently fail, don't show error to user
            } finally {
                faceDetecting.value = false;
            }
        };

        // Update role name for a detected face
        const updateFaceRoleName = (faceIndex, roleName) => {
            const currentForm = getCurrentForm();
            if (currentForm && currentForm.detectedFaces && currentForm.detectedFaces[faceIndex]) {
                // 使用展开运算符创建新对象，确保响应式更新
                currentForm.detectedFaces[faceIndex] = {
                    ...currentForm.detectedFaces[faceIndex],
                    roleName: roleName
                };
                // 触发响应式更新
                currentForm.detectedFaces = [...currentForm.detectedFaces];
            }
        };

        // Toggle editing state for a face
        const toggleFaceEditing = (faceIndex) => {
            const currentForm = getCurrentForm();
            if (currentForm && currentForm.detectedFaces && currentForm.detectedFaces[faceIndex]) {
                // 使用展开运算符创建新对象，确保响应式更新
                currentForm.detectedFaces[faceIndex] = {
                    ...currentForm.detectedFaces[faceIndex],
                    isEditing: !currentForm.detectedFaces[faceIndex].isEditing
                };
                // 触发响应式更新
                currentForm.detectedFaces = [...currentForm.detectedFaces];
            }
        };

        // Save face role name and exit editing
        const saveFaceRoleName = (faceIndex, roleName) => {
            const currentForm = getCurrentForm();
            if (currentForm && currentForm.detectedFaces && currentForm.detectedFaces[faceIndex]) {
                // 同时更新 roleName 和 isEditing，确保响应式更新
                currentForm.detectedFaces[faceIndex] = {
                    ...currentForm.detectedFaces[faceIndex],
                    roleName: roleName || currentForm.detectedFaces[faceIndex].roleName,
                    isEditing: false
                };
                // 触发响应式更新
                currentForm.detectedFaces = [...currentForm.detectedFaces];
            }
            // 同步更新所有关联的音频播放器角色名
            // 只有当任务类型是 s2v 且有分离的音频时才需要更新
            if (selectedTaskId.value === 's2v' && s2vForm.value.separatedAudios) {
                s2vForm.value.separatedAudios.forEach((audio, index) => {
                    // 如果音频的 roleIndex 等于当前修改的 faceIndex，则更新其 roleName
                    if (audio.roleIndex === faceIndex) {
                        s2vForm.value.separatedAudios[index].roleName = roleName || `角色${faceIndex + 1}`;
                    }
                });
                // 使用展开运算符确保响应式更新
                s2vForm.value.separatedAudios = [...s2vForm.value.separatedAudios];
            }
        };

        const selectTask = (taskType) => {
            console.log('[selectTask] 开始切换任务类型:', {
                taskType,
                currentSelectedTaskId: selectedTaskId.value,
                currentSelectedModel: selectedModel.value,
                currentFormModel: getCurrentForm().model_cls,
                currentFormStage: getCurrentForm().stage
            });

            for (const t of models.value.map(m => m.task)) {
                if (getTaskTypeName(t) === taskType) {
                    taskType = t;
                }
            }
            selectedTaskId.value = taskType;

            console.log('[selectTask] 任务类型已更新:', {
                newTaskType: selectedTaskId.value,
                availableModels: models.value.filter(m => m.task === taskType)
            });

            // 根据任务类型恢复对应的预览
            if (taskType === 'i2v' && i2vForm.value.imageFile) {
                // 恢复图片预览
                const reader = new FileReader();
                reader.onload = (e) => {
                    setCurrentImagePreview(e.target.result);
                };
                reader.readAsDataURL(i2vForm.value.imageFile);
            } else if (taskType === 's2v') {
                // 恢复数字人任务的图片和音频预览
                if (s2vForm.value.imageFile) {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        setCurrentImagePreview(e.target.result);
                    };
                    reader.readAsDataURL(s2vForm.value.imageFile);
                }
                if (s2vForm.value.audioFile) {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        setCurrentAudioPreview(e.target.result);
                    };
                    reader.readAsDataURL(s2vForm.value.audioFile);
                }
            } else if (taskType === 'animate') {
                // 恢复角色替换任务的图片和视频预览
                if (animateForm.value.imageFile) {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        setCurrentImagePreview(e.target.result);
                    };
                    reader.readAsDataURL(animateForm.value.imageFile);
                }
                if (animateForm.value.videoFile) {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        setCurrentVideoPreview(e.target.result);
                    };
                    reader.readAsDataURL(animateForm.value.videoFile);
                }
                // 确保 animate 任务类型有默认 prompt
                if (!animateForm.value.prompt || animateForm.value.prompt.trim() === '') {
                    animateForm.value.prompt = '视频中的人在做动作';
                }
            }

            // 自动选择该任务类型下的第一个模型（仅在当前模型无效且 URL 中没有 model 参数时）
            const currentForm = getCurrentForm();
            const urlParams = new URLSearchParams(window.location.search);
            const hasModelInUrl = urlParams.has('model');

            // 获取新任务类型下的可用模型
            const availableModels = models.value.filter(m => m.task === taskType);

            if (availableModels.length > 0) {
                // 检查当前选择的模型和stage是否属于新任务类型
                const currentModel = currentForm.model_cls || selectedModel.value;
                const currentStage = currentForm.stage;
                const isCurrentModelValid = currentModel && availableModels.some(m =>
                    m.model_cls === currentModel && m.stage === currentStage
                );

                // 如果当前模型无效且 URL 中没有 model 参数，自动选择第一个模型
                if (!isCurrentModelValid || !hasModelInUrl) {
                    const firstModel = availableModels[0];
                    console.log('[selectTask] 自动选择第一个模型:', {
                        firstModel: firstModel.model_cls,
                        firstStage: firstModel.stage,
                        reason: !isCurrentModelValid ? '当前模型或stage无效' : 'URL中没有model参数',
                        currentModel,
                        currentStage
                    });
                    // 直接调用 selectModel 来确保路由也会更新
                    selectModel(firstModel.model_cls);
                } else {
                    console.log('[selectTask] 不自动选择模型:', {
                        isCurrentModelValid,
                        hasModelInUrl,
                        currentModel,
                        currentStage
                    });
                }
            }
        };

        const selectModel = (model) => {
            console.log('[selectModel] 开始切换模型:', {
                model,
                currentSelectedModel: selectedModel.value,
                currentTaskType: selectedTaskId.value,
                currentFormModel: getCurrentForm().model_cls,
                currentFormStage: getCurrentForm().stage
            });

            selectedModel.value = model;
            const currentForm = getCurrentForm();
            currentForm.model_cls = model;
            // 自动设置 stage 为模型对应的第一个 stage
            const availableStages = models.value
                .filter(m => m.task === selectedTaskId.value && m.model_cls === model)
                .map(m => m.stage);
            if (availableStages.length > 0) {
                currentForm.stage = availableStages[0];
                console.log('[selectModel] 自动设置 stage:', {
                    stage: currentForm.stage,
                    availableStages
                });
            }

            console.log('[selectModel] 模型切换完成:', {
                selectedModel: selectedModel.value,
                formModel: currentForm.model_cls,
                formStage: currentForm.stage
            });
        };

        const triggerImageUpload = () => {
            document.querySelector('input[type="file"][accept="image/*"]').click();
        };

        const triggerAudioUpload = () => {
            const audioInput = document.querySelector('input[type="file"][data-role="audio-input"]');
            if (audioInput) {
                audioInput.click();
            } else {
                console.warn('音频输入框未找到');
            }
        };

        const removeImage = () => {
            setCurrentImagePreview(null);
            if (selectedTaskId.value === 'i2v') {
                i2vForm.value.imageFile = null;
                i2vForm.value.detectedFaces = [];
            } else if (selectedTaskId.value === 's2v') {
                s2vForm.value.imageFile = null;
                s2vForm.value.detectedFaces = [];
            } else if (selectedTaskId.value === 'animate') {
                animateForm.value.imageFile = null;
                animateForm.value.detectedFaces = [];
            }
            updateUploadedContentStatus();
            // 重置文件输入框，确保可以重新选择相同文件
            const imageInput = document.querySelector('input[type="file"][accept="image/*"]');
            if (imageInput) {
                imageInput.value = '';
            }
        };

        const removeAudio = () => {
            setCurrentAudioPreview(null);
            s2vForm.value.audioFile = null;
            s2vForm.value.separatedAudios = [];
            updateUploadedContentStatus();
            console.log('音频已移除');
            // 重置音频文件输入框，确保可以重新选择相同文件
            const audioInput = document.querySelector('input[type="file"][data-role="audio-input"]');
            if (audioInput) {
                audioInput.value = '';
            }
        };

        // 删除视频（用于 animate 任务类型）
        const removeVideo = () => {
            setCurrentVideoPreview(null);
            animateForm.value.videoFile = null;
            updateUploadedContentStatus();
            console.log('视频已移除');
            // 重置视频文件输入框，确保可以重新选择相同文件
            const videoInput = document.querySelector('input[type="file"][data-role="video-input"]');
            if (videoInput) {
                videoInput.value = '';
            }
        };

        // Update role assignment for separated audio
        const updateSeparatedAudioRole = (speakerIndex, roleIndex) => {
            if (s2vForm.value.separatedAudios && s2vForm.value.separatedAudios[speakerIndex]) {
                const currentForm = getCurrentForm();
                const detectedFaces = currentForm?.detectedFaces || [];

                if (roleIndex >= 0 && roleIndex < detectedFaces.length) {
                    s2vForm.value.separatedAudios[speakerIndex].roleName = detectedFaces[roleIndex].roleName || `角色${roleIndex + 1}`;
                    s2vForm.value.separatedAudios[speakerIndex].roleIndex = roleIndex;
                }
            }
        };

        // Update audio name for a separated audio
        const updateSeparatedAudioName = (audioIndex, audioName) => {
            if (s2vForm.value.separatedAudios && s2vForm.value.separatedAudios[audioIndex]) {
                s2vForm.value.separatedAudios[audioIndex].audioName = audioName;
            }
        };

        // Toggle editing state for a separated audio
        const toggleSeparatedAudioEditing = (audioIndex) => {
            if (s2vForm.value.separatedAudios && s2vForm.value.separatedAudios[audioIndex]) {
                s2vForm.value.separatedAudios[audioIndex].isEditing = !s2vForm.value.separatedAudios[audioIndex].isEditing;
            }
        };

        // Save separated audio name and exit editing
        const saveSeparatedAudioName = (audioIndex, audioName) => {
            updateSeparatedAudioName(audioIndex, audioName);
            toggleSeparatedAudioEditing(audioIndex);
        };

        const getAudioMimeType = () => {
            if (s2vForm.value.audioFile) {
                return s2vForm.value.audioFile.type;
            }
            return 'audio/mpeg'; // 默认类型
        };

        // Separate audio tracks for multiple speakers
        const separateAudioTracks = async (audioDataUrl, numSpeakers) => {
            audioSeparating.value = true;  // 开始音频分割，显示加载状态
            try {
                // 优先使用 audioFile（如果存在），因为它包含完整的文件信息，避免 data URL 格式问题
                const currentForm = getCurrentForm();
                let audioData = audioDataUrl;

                if (currentForm?.audioFile && currentForm.audioFile instanceof File) {
                    // 使用 File 对象，读取为 base64，确保格式正确
                    try {
                        const fileDataUrl = await new Promise((resolve, reject) => {
                            const reader = new FileReader();
                            reader.onload = () => resolve(reader.result);
                            reader.onerror = reject;
                            reader.readAsDataURL(currentForm.audioFile);
                        });
                        audioData = fileDataUrl;
                        console.log('Using audioFile for separation, format:', currentForm.audioFile.type);
                    } catch (error) {
                        console.warn('Failed to read audioFile, falling back to audioDataUrl:', error);
                        // 如果读取失败，继续使用 audioDataUrl
                    }
                }

                // Clean and validate base64 string before sending
                let cleanedAudioData = audioData;
                if (audioData.includes(',')) {
                    // If it's a data URL, extract the base64 part
                    const parts = audioData.split(',');
                    if (parts.length > 1) {
                        cleanedAudioData = parts.slice(1).join(','); // Join in case there are multiple commas
                    }
                }

                // Remove any whitespace and newlines
                cleanedAudioData = cleanedAudioData.trim().replace(/\s/g, '');

                // Check if it's a valid base64 string length (must be multiple of 4)
                const missingPadding = cleanedAudioData.length % 4;
                if (missingPadding !== 0) {
                    console.warn(`[separateAudioTracks] Base64 string length (${cleanedAudioData.length}) is not a multiple of 4, adding padding`);
                    cleanedAudioData += '='.repeat(4 - missingPadding);
                }

                // Reconstruct data URL if it was originally a data URL
                if (audioData.startsWith('data:')) {
                    const header = audioData.split(',')[0];
                    cleanedAudioData = `${header},${cleanedAudioData}`;
                }

                console.log(`[separateAudioTracks] Sending audio for separation, length: ${cleanedAudioData.length}, num_speakers: ${numSpeakers}`);

                const response = await apiCall('/api/v1/audio/separate', {
                    method: 'POST',
                    body: JSON.stringify({
                        audio: cleanedAudioData,
                        num_speakers: numSpeakers
                    })
                });

                if (!response.ok) {
                    console.error('Audio separation failed:', response.status, response.statusText);
                    audioSeparating.value = false;
                    return;
                }

                const data = await response.json();
                console.log('Audio separation response:', data);

                if (data && data.speakers && data.speakers.length > 0) {
                    const currentForm = getCurrentForm();
                    const detectedFaces = currentForm?.detectedFaces || [];

                    // Map separated speakers to detected faces
                    // Initialize with first role if available
                    const separatedAudios = data.speakers.map((speaker, index) => {
                        const faceIndex = index < detectedFaces.length ? index : 0;
                        return {
                            speaker_id: speaker.speaker_id,
                            audio: speaker.audio,  // Base64 encoded audio
                            audioDataUrl: `data:audio/wav;base64,${speaker.audio}`,  // Data URL for preview
                            audioName: `音色${index + 1}`,  // 音频名称，默认显示为"音色1"、"音色2"等
                            roleName: detectedFaces[faceIndex]?.roleName || `角色${faceIndex + 1}`,  // 关联的角色名称
                            roleIndex: faceIndex,
                            isEditing: false,  // 编辑状态
                            sample_rate: speaker.sample_rate,
                            segments: speaker.segments
                        };
                    });

                    // Update separatedAudios and trigger reactivity
                    s2vForm.value.separatedAudios = [...separatedAudios];  // Use spread to ensure reactivity
                    console.log('Updated separatedAudios:', s2vForm.value.separatedAudios.length, 'speakers', s2vForm.value.separatedAudios);
                } else {
                    console.warn('No speakers found in separation response:', data);
                    s2vForm.value.separatedAudios = [];
                }
                audioSeparating.value = false;  // 音频分割完成，隐藏加载状态
            } catch (error) {
                console.error('Audio separation error:', error);
                audioSeparating.value = false;  // 发生错误时也要隐藏加载状态
                throw error;
            }
        };

        const handleAudioUpload = async (event) => {
            const file = event.target.files[0];

            if (file && (file.type?.startsWith('audio/') || file.type?.startsWith('video/'))) {
                const allowedVideoTypes = ['video/mp4', 'video/x-m4v', 'video/mpeg'];
                if (file.type.startsWith('video/') && !allowedVideoTypes.includes(file.type)) {
                    showAlert(t('unsupportedVideoFormat'), 'warning');
                    setCurrentAudioPreview(null);
                    s2vForm.value.separatedAudios = [];
                    updateUploadedContentStatus();
                    return;
                }
                s2vForm.value.audioFile = file;

                // Read file as data URL for preview
                const reader = new FileReader();
                reader.onload = async (e) => {
                    const audioDataUrl = e.target.result;
                    setCurrentAudioPreview(audioDataUrl);
                    updateUploadedContentStatus();
                    // 音频分离由统一的 watch 监听器处理，不需要在这里手动调用
                    console.log('[handleAudioUpload] 音频上传完成，音频分离将由统一的监听器自动处理');
                };
                reader.readAsDataURL(file);
            } else {
                setCurrentAudioPreview(null);
                s2vForm.value.separatedAudios = [];
                updateUploadedContentStatus();
                if (file) {
                    showAlert(t('unsupportedAudioOrVideo'), 'warning');
                }
            }
        };

        // 处理视频上传（用于 animate 任务类型）
        const handleVideoUpload = async (event) => {
            const file = event.target.files[0];

            if (file && file.type?.startsWith('video/')) {
                const allowedVideoTypes = ['video/mp4', 'video/x-m4v', 'video/mpeg', 'video/webm', 'video/quicktime'];
                if (!allowedVideoTypes.includes(file.type)) {
                    showAlert(t('unsupportedVideoFormat') || '不支持的视频格式', 'warning');
                    setCurrentVideoPreview(null);
                    animateForm.value.videoFile = null;
                    updateUploadedContentStatus();
                    return;
                }
                animateForm.value.videoFile = file;

                // Read file as data URL for preview
                const reader = new FileReader();
                reader.onload = async (e) => {
                    const videoDataUrl = e.target.result;
                    setCurrentVideoPreview(videoDataUrl);
                    updateUploadedContentStatus();
                };
                reader.readAsDataURL(file);
            } else {
                setCurrentVideoPreview(null);
                animateForm.value.videoFile = null;
                updateUploadedContentStatus();
                if (file) {
                    showAlert(t('unsupportedVideoFormat') || '不支持的视频格式', 'warning');
                }
            }
        };

        // 开始录音
        const startRecording = async () => {
            try {
                console.log('开始录音...');

                // 检查浏览器支持
                if (!navigator.mediaDevices) {
                    throw new Error('该浏览器不支持录音功能');
                }

                if (!navigator.mediaDevices.getUserMedia) {
                    throw new Error('浏览器不支持录音功能，请确保使用HTTPS协议访问');
                }

                if (!window.MediaRecorder) {
                    throw new Error('浏览器不支持MediaRecorder，请更新到最新版本浏览器');
                }

                // 检查HTTPS协议
                console.log('当前协议:', location.protocol, '主机名:', location.hostname);
                if (location.protocol !== 'https:' && location.hostname !== 'localhost' && !location.hostname.includes('127.0.0.1')) {
                    throw new Error(`录音功能需要HTTPS协议，当前使用${location.protocol}协议。请使用HTTPS访问网站或通过localhost:端口号访问`);
                }

                console.log('浏览器支持检查通过，请求麦克风权限...');

                // 记录浏览器支持状态用于调试
                const browserSupport = {
                    mediaDevices: !!navigator.mediaDevices,
                    getUserMedia: !!navigator.mediaDevices?.getUserMedia,
                    MediaRecorder: !!window.MediaRecorder,
                    protocol: location.protocol,
                    hostname: location.hostname,
                    userAgent: navigator.userAgent
                };
                console.log('浏览器支持状态:', browserSupport);

                // 请求麦克风权限
                console.log('正在请求麦克风权限...');
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        sampleRate: 44100
                    }
                });
                console.log('麦克风权限获取成功，音频流:', stream);

                // 创建MediaRecorder
                mediaRecorder.value = new MediaRecorder(stream, {
                    mimeType: 'audio/webm;codecs=opus'
                });

                audioChunks.value = [];

                // 监听数据可用事件
                mediaRecorder.value.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.value.push(event.data);
                    }
                };

                // 监听录音停止事件
                mediaRecorder.value.onstop = () => {
                    const audioBlob = new Blob(audioChunks.value, { type: 'audio/webm' });
                    const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' });

                    // 设置到表单
                    s2vForm.value.audioFile = audioFile;

                    // 创建预览URL
                    const audioUrl = URL.createObjectURL(audioBlob);
                    setCurrentAudioPreview(audioUrl);
                    updateUploadedContentStatus();

                    // 停止所有音频轨道
                    stream.getTracks().forEach(track => track.stop());

                    showAlert(t('recordingCompleted'), 'success');
                };

                // 开始录音
                mediaRecorder.value.start(1000); // 每秒收集一次数据
                isRecording.value = true;
                recordingDuration.value = 0;

                // 开始计时
                recordingTimer.value = setInterval(() => {
                    recordingDuration.value++;
                }, 1000);

                showAlert(t('recordingStarted'), 'info');

            } catch (error) {
                console.error('录音失败:', error);
                let errorMessage = t('recordingFailed');

                if (error.name === 'NotAllowedError') {
                    errorMessage = t('microphonePermissionDenied');
                } else if (error.name === 'NotFoundError') {
                    errorMessage = t('microphoneNotFound');
                } else if (error.name === 'NotSupportedError') {
                    errorMessage = t('recordingNotSupportedOnMobile');
                } else if (error.name === 'NotReadableError') {
                    errorMessage = t('microphoneInUse');
                } else if (error.name === 'OverconstrainedError') {
                    errorMessage = t('microphoneNotCompatible');
                } else if (error.name === 'SecurityError') {
                    errorMessage = t('securityErrorUseHttps');
                } else if (error.message) {
                    errorMessage = error.message;
                }

                // 添加调试信息
                const debugInfo = {
                    userAgent: navigator.userAgent,
                    protocol: location.protocol,
                    hostname: location.hostname,
                    mediaDevices: !!navigator.mediaDevices,
                    getUserMedia: !!navigator.mediaDevices?.getUserMedia,
                    MediaRecorder: !!window.MediaRecorder,
                    isSecureContext: window.isSecureContext,
                    chromeVersion: navigator.userAgent.match(/Chrome\/(\d+)/)?.[1] || '未知'
                };
                console.log('浏览器调试信息:', debugInfo);

                // 如果是Chrome但仍有问题，提供特定建议
                if (navigator.userAgent.includes('Chrome')) {
                    console.log('检测到Chrome浏览器，可能的问题:');
                    console.log('1. 请确保使用HTTPS协议或localhost访问');
                    console.log('2. 检查Chrome地址栏是否有麦克风权限');
                    console.log('3. 尝试在Chrome设置中重置网站权限');
                    console.log('4. 确保没有其他应用占用麦克风');
                }

                showAlert(errorMessage, 'danger');
            }
        };

        // 停止录音
        const stopRecording = () => {
            if (mediaRecorder.value && isRecording.value) {
                mediaRecorder.value.stop();
                isRecording.value = false;

                if (recordingTimer.value) {
                    clearInterval(recordingTimer.value);
                    recordingTimer.value = null;
                }

                showAlert(t('recordingStopped'), 'info');
            }
        };

        // 格式化录音时长
        const formatRecordingDuration = (seconds) => {
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        };

        const submitTask = async () => {
            try {
                // 检查是否正在加载模板
                if (templateLoading.value) {
                    showAlert(t('templateLoadingPleaseWait'), 'warning');
                    return;
                }

                const currentForm = getCurrentForm();

                // 表单验证
                if (!selectedTaskId.value) {
                    showAlert(t('pleaseSelectTaskType'), 'warning');
                    return;
                }

                if (!currentForm.model_cls) {
                    showAlert(t('pleaseSelectModel'), 'warning');
                    return;
                }

                // animate 任务类型不需要 prompt，其他任务类型需要
                if (selectedTaskId.value !== 'animate') {
                    if (!currentForm.prompt || currentForm.prompt.trim().length === 0) {
                        if (selectedTaskId.value === 's2v') {
                            currentForm.prompt = '让角色根据音频内容自然说话';
                        } else {
                            showAlert(t('pleaseEnterPrompt'), 'warning');
                            return;
                        }
                    }

                    if (currentForm.prompt.length > 1000) {
                        showAlert(t('promptTooLong'), 'warning');
                        return;
                    }
                }

                if (selectedTaskId.value === 'i2v' && !currentForm.imageFile) {
                    showAlert(t('i2vTaskRequiresImage'), 'warning');
                    return;
                }

                if (selectedTaskId.value === 's2v' && !currentForm.imageFile) {
                    showAlert(t('s2vTaskRequiresImage'), 'warning');
                    return;
                }

                if (selectedTaskId.value === 's2v' && !currentForm.audioFile) {
                    showAlert(t('s2vTaskRequiresAudio'), 'warning');
                    return;
                }

                if (selectedTaskId.value === 'animate' && !currentForm.imageFile) {
                    showAlert(t('animateTaskRequiresImage'), 'warning');
                    return;
                }

                if (selectedTaskId.value === 'animate' && !currentForm.videoFile) {
                    showAlert(t('animateTaskRequiresVideo'), 'warning');
                    return;
                }
                submitting.value = true;

                // 确定实际提交的任务类型
                let actualTaskType = selectedTaskId.value;

                var formData = {
                    task: actualTaskType,
                    model_cls: currentForm.model_cls,
                    stage: currentForm.stage,
                    seed: currentForm.seed || Math.floor(Math.random() * 1000000)
                };

                // animate 任务类型使用默认 prompt，其他任务类型需要用户输入
                if (selectedTaskId.value === 'animate') {
                    // animate 任务类型使用默认 prompt
                    formData.prompt = currentForm.prompt && currentForm.prompt.trim().length > 0
                        ? currentForm.prompt.trim()
                        : '视频中的人在做动作';
                } else {
                    formData.prompt = currentForm.prompt ? currentForm.prompt.trim() : '';
                }

                if (currentForm.model_cls.startsWith('wan2.1')) {
                    formData.negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
                }

                if (selectedTaskId.value === 'i2v' && currentForm.imageFile) {
                    const base64 = await fileToBase64(currentForm.imageFile);
                    formData.input_image = {
                        type: 'base64',
                        data: base64
                    };
                }

                if (selectedTaskId.value === 'animate' && currentForm.imageFile) {
                    const base64 = await fileToBase64(currentForm.imageFile);
                    formData.input_image = {
                        type: 'base64',
                        data: base64
                    };
                }

                if (selectedTaskId.value === 'animate' && currentForm.videoFile) {
                    const base64 = await fileToBase64(currentForm.videoFile);
                    formData.input_video = {
                        type: 'base64',
                        data: base64
                    };
                }

                if (selectedTaskId.value === 's2v') {
                    if (currentForm.imageFile) {
                        const base64 = await fileToBase64(currentForm.imageFile);
                        formData.input_image = {
                            type: 'base64',
                            data: base64
                        };
                    }

                    // 检测是否为多人模式：有多个分离的音频和多个角色
                    const isMultiPersonMode = s2vForm.value.separatedAudios &&
                                            s2vForm.value.separatedAudios.length > 1 &&
                                            currentForm.detectedFaces &&
                                            currentForm.detectedFaces.length > 1;

                    if (isMultiPersonMode) {
                        // 多人模式：生成mask图、保存音频文件、生成config.json
                        try {
                            const multiPersonData = await prepareMultiPersonAudio(
                                currentForm.detectedFaces,
                                s2vForm.value.separatedAudios,
                                currentForm.imageFile,
                                currentForm.audioFile  // 传递原始音频文件
                            );

                            formData.input_audio = {
                                type: 'directory',
                                data: multiPersonData
                            };
                            formData.negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
                        } catch (error) {
                            console.error('Failed to prepare multi-person audio:', error);
                            showAlert(t('prepareMultiPersonAudioFailed') + ': ' + error.message, 'danger');
                            submitting.value = false;
                            return;
                        }
                    } else if (currentForm.audioFile) {
                        // 单人模式：使用原始音频文件
                        const base64 = await fileToBase64(currentForm.audioFile);
                        formData.input_audio = {
                            type: 'base64',
                            data: base64
                        };
                        formData.negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
                    }
                }

                const response = await apiRequest('/api/v1/task/submit', {
                    method: 'POST',
                    body: JSON.stringify(formData)
                });

                if (response && response.ok) {
                    let result;
                    try {
                        result = await response.json();
                    } catch (error) {
                        console.error('Failed to parse response JSON:', error);
                        showAlert(t('taskSubmittedButParseFailed'), 'warning');
                        submitting.value = false;
                        return null;
                    }

                    showAlert(t('taskSubmitSuccessAlert'), 'success');

                    // 开始轮询新提交的任务状态（不等待，异步执行）
                    try {
                        startPollingTask(result.task_id);
                    } catch (error) {
                        console.error('Failed to start polling task:', error);
                        // 不阻止流程继续
                    }

                    // 保存完整的任务历史（包括提示词、图片和音频）- 异步执行，不阻塞
                    // 注意：addTaskToHistory 是同步函数，但为了统一处理，使用 Promise.resolve 包装
                    Promise.resolve().then(() => {
                        try {
                            addTaskToHistory(selectedTaskId.value, currentForm);
                        } catch (error) {
                            console.error('Failed to add task to history:', error);
                        }
                    }).catch(error => {
                        console.error('Failed to add task to history:', error);
                    });

                    // 重置表单（异步执行，不阻塞）- 使用 Promise.race 添加超时保护
                    try {
                        await Promise.race([
                            Promise.resolve(resetForm(selectedTaskId.value)),
                            new Promise((_, reject) => setTimeout(() => reject(new Error('resetForm timeout')), 3000))
                        ]);
                    } catch (error) {
                        console.error('Failed to reset form:', error);
                        // 不阻止流程继续，只记录错误
                    }

                    // 重置当前任务类型的表单（保留模型选择，清空图片、音频和提示词）
                    try {
                        selectedTaskId.value = selectedTaskId.value;
                        selectModel(currentForm.model_cls);
                    } catch (error) {
                        console.error('Failed to select model:', error);
                        // 不阻止流程继续
                    }

                    // 返回新创建的任务ID
                    return result.task_id;
                } else {
                    let error;
                    try {
                        error = await response.json();
                        showAlert(`${t('taskSubmitFailedAlert')}: ${error.message || 'Unknown error'},${error.detail || ''}`, 'danger');
                    } catch (parseError) {
                        console.error('Failed to parse error response:', parseError);
                        showAlert(`${t('taskSubmitFailedAlert')}: ${response.statusText || 'Unknown error'}`, 'danger');
                    }
                    return null;
                }
            } catch (error) {
                showAlert(`${t('submitTaskFailedAlert')}: ${error.message}`, 'danger');
                return null;
            } finally {
                submitting.value = false;
            }
        };

        const fileToBase64 = (file) => {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onload = () => {
                    const base64 = reader.result.split(',')[1];
                    resolve(base64);
                };
                reader.onerror = error => reject(error);
            });
        };

        // 准备多人模式的音频数据：生成mask图、保存音频文件、生成config.json
        const prepareMultiPersonAudio = async (detectedFaces, separatedAudios, imageFile, originalAudioFile) => {
            // 1. 读取原始图片，获取尺寸
            const imageBase64 = await fileToBase64(imageFile);
            const imageDataUrl = `data:image/png;base64,${imageBase64}`;

            // 创建图片对象以获取尺寸
            const img = new Image();
            await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = reject;
                img.src = imageDataUrl;
            });
            const imageWidth = img.naturalWidth;
            const imageHeight = img.naturalHeight;

            // 2. 为每个角色生成mask图和音频文件
            const directoryFiles = {};
            const talkObjects = [];

            for (let i = 0; i < detectedFaces.length; i++) {
                const face = detectedFaces[i];
                const audioIndex = i < separatedAudios.length ? i : 0;
                const audio = separatedAudios[audioIndex];

                // 生成mask图（box部分为白色，其余部分为黑色）
                const maskBase64 = await generateMaskImage(
                    face.bbox,
                    imageWidth,
                    imageHeight
                );

                // 保存mask图
                const maskFilename = `p${i + 1}_mask.png`;
                directoryFiles[maskFilename] = maskBase64;

                // 保存音频文件
                // 注意：separatedAudios中的audio已经是base64编码的wav格式
                const audioFilename = `p${i + 1}.wav`;
                directoryFiles[audioFilename] = audio.audio; // audio.audio是base64编码的wav数据

                // 添加到talk_objects
                talkObjects.push({
                    audio: audioFilename,
                    mask: maskFilename
                });
            }

            // 3. 保存原始未分割的音频文件（用于后续复用）
            if (originalAudioFile) {
                try {
                    // 将原始音频文件转换为base64
                    const originalAudioBase64 = await fileToBase64(originalAudioFile);
                    // 根据原始文件名确定扩展名，如果没有扩展名则使用.wav
                    const originalFilename = originalAudioFile.name || 'original_audio.wav';
                    const fileExtension = originalFilename.toLowerCase().split('.').pop();
                    const validExtensions = ['wav', 'mp3', 'mp4', 'aac', 'ogg', 'm4a'];
                    const extension = validExtensions.includes(fileExtension) ? fileExtension : 'wav';
                    const originalAudioFilename = `original_audio.${extension}`;
                    directoryFiles[originalAudioFilename] = originalAudioBase64;
                    console.log('已保存原始音频文件:', originalAudioFilename);
                } catch (error) {
                    console.warn('保存原始音频文件失败:', error);
                    // 不阻止任务提交，只记录警告
                }
            }

            // 4. 生成config.json
            const configJson = {
                talk_objects: talkObjects
            };
            const configJsonString = JSON.stringify(configJson, null, 4);
            const configBase64 = btoa(unescape(encodeURIComponent(configJsonString)));
            directoryFiles['config.json'] = configBase64;

            return directoryFiles;
        };

        // 生成mask图：根据bbox坐标生成白色区域，其余为黑色
        const generateMaskImage = async (bbox, imageWidth, imageHeight) => {
            // bbox格式: [x1, y1, x2, y2]
            const [x1, y1, x2, y2] = bbox;

            // 创建canvas
            const canvas = document.createElement('canvas');
            canvas.width = imageWidth;
            canvas.height = imageHeight;
            const ctx = canvas.getContext('2d');

            // 填充黑色背景
            ctx.fillStyle = '#000000';
            ctx.fillRect(0, 0, imageWidth, imageHeight);

            // 在bbox区域填充白色
            ctx.fillStyle = '#FFFFFF';
            ctx.fillRect(Math.round(x1), Math.round(y1), Math.round(x2 - x1), Math.round(y2 - y1));

            // 转换为base64
            return canvas.toDataURL('image/png').split(',')[1];
        };

        const formatTime = (timestamp) => {
            if (!timestamp) return '';
            const date = new Date(timestamp * 1000);
            return date.toLocaleString('zh-CN');
        };

        // 通用缓存管理函数
        const loadFromCache = (cacheKey, expiryKey) => {
            try {
                const cached = localStorage.getItem(cacheKey);
                if (cached) {
                    const data = JSON.parse(cached);
                    if (Date.now() - data.timestamp < expiryKey) {
                        console.log(`成功从缓存加载数据${cacheKey}:`, data.data);
                        return data.data;
                    } else {
                        // 缓存过期，清除
                        localStorage.removeItem(cacheKey);
                        console.log(`缓存过期，清除 ${cacheKey}`);
                    }
                }
            } catch (error) {
                console.warn(`加载缓存失败 ${cacheKey}:`, error);
                localStorage.removeItem(cacheKey);
            }
            return null;
        };

        const saveToCache = (cacheKey, data) => {
            try {
                const cacheData = {
                    data: data,
                    timestamp: Date.now()
                };
                console.log(`成功保存缓存数据 ${cacheKey}:`, cacheData);
                localStorage.setItem(cacheKey, JSON.stringify(cacheData));
            } catch (error) {
                console.warn(`保存缓存失败 ${cacheKey}:`, error);
            }
        };

        // 清除所有应用缓存
        const clearAllCache = () => {
            try {
                const cacheKeys = [
                    TASK_FILE_CACHE_KEY,
                    TEMPLATE_FILE_CACHE_KEY,
                    MODELS_CACHE_KEY,
                    TEMPLATES_CACHE_KEY
                ];

                // 清除所有任务缓存（使用通配符匹配）
                for (let i = 0; i < localStorage.length; i++) {
                    const key = localStorage.key(i);
                    if (key && key.startsWith(TASKS_CACHE_KEY)) {
                        localStorage.removeItem(key);
                    }
                }

                // 清除所有模板缓存（使用通配符匹配）
                for (let i = 0; i < localStorage.length; i++) {
                    const key = localStorage.key(i);
                    if (key && key.startsWith(TEMPLATES_CACHE_KEY)) {
                        localStorage.removeItem(key);
                    }
                }
                // 清除其他缓存
                cacheKeys.forEach(key => {
                    localStorage.removeItem(key);
                });

                // 清除内存中的任务文件缓存
                taskFileCache.value.clear();
                taskFileCacheLoaded.value = false;

                // 清除内存中的模板文件缓存
                templateFileCache.value.clear();
                templateFileCacheLoaded.value = false;

                console.log('所有缓存已清除');
            } catch (error) {
                console.warn('清除缓存失败:', error);
            }
        };

        // 模板文件缓存管理函数
        const loadTemplateFilesFromCache = () => {
            try {
                const cached = localStorage.getItem(TEMPLATE_FILE_CACHE_KEY);
                if (cached) {
                    const data = JSON.parse(cached);
                    if (data.files) {
                        for (const [cacheKey, fileData] of Object.entries(data.files)) {
                            templateFileCache.value.set(cacheKey, fileData);
                        }
                        return true;
                    } else {
                        console.warn('模板文件缓存数据格式错误');
                        return false;
                    }
                }
            } catch (error) {
                console.warn('加载模板文件缓存失败:', error);
            }
            return false;
        };

        const saveTemplateFilesToCache = () => {
            try {
                const files = {};
                for (const [cacheKey, fileData] of templateFileCache.value.entries()) {
                    files[cacheKey] = fileData;
                }
                const data = {
                    files: files,
                    timestamp: Date.now()
                };
                localStorage.setItem(TEMPLATE_FILE_CACHE_KEY, JSON.stringify(data));
            } catch (error) {
                console.warn('保存模板文件缓存失败:', error);
            }
        };

        const getTemplateFileCacheKey = (templateId, fileKey) => {
            return `template_${templateId}_${fileKey}`;
        };

        const getTemplateFileFromCache = (cacheKey) => {
            return templateFileCache.value.get(cacheKey) || null;
        };

        const setTemplateFileToCache = (fileKey, fileData) => {
            templateFileCache.value.set(fileKey, fileData);
            // 异步保存到localStorage
            setTimeout(() => {
                saveTemplateFilesToCache();
            }, 100);
        };

        const getTemplateFileUrlFromApi = async (fileKey, fileType) => {
            const apiUrl = `/api/v1/template/asset_url/${fileType}/${fileKey}`;
            const response = await apiRequest(apiUrl);
            if (response && response.ok) {
                const data = await response.json();
                let assertUrl = data.url;
                if (assertUrl.startsWith('./assets/')) {
                    const token = localStorage.getItem('accessToken');
                    if (token) {
                        assertUrl = `${assertUrl}&token=${encodeURIComponent(token)}`;
                    }
                }
                setTemplateFileToCache(fileKey, {
                    url: assertUrl,
                    timestamp: Date.now()
                });
                return assertUrl;
            }
            return null;
        };

        // 获取模板文件URL（优先从缓存，缓存没有则生成URL）- 同步版本
        const getTemplateFileUrl = (fileKey, fileType) => {
            // 检查参数有效性（静默处理，不打印警告，因为模板可能确实没有某些输入）
            if (!fileKey) {
                return null;
            }

            // 先从缓存获取
            const cachedFile = getTemplateFileFromCache(fileKey);
            if (cachedFile) {
                /* console.log('从缓存获取模板文件url', { fileKey});*/
                return cachedFile.url;
            }
            // 如果缓存中没有，返回null，让调用方知道需要异步获取
            console.warn('模板文件URL不在缓存中，需要异步获取:', { fileKey, fileType });
            getTemplateFileUrlAsync(fileKey, fileType).then(url => {
                return url;
            });
            return null;
        };

        // 创建响应式的模板文件URL（用于首屏渲染）
        const createTemplateFileUrlRef = (fileKey, fileType) => {
            const urlRef = ref(null);

            // 检查参数有效性（静默处理，不打印警告）
            if (!fileKey) {
                return urlRef;
            }

            // 先从缓存获取
            const cachedFile = getTemplateFileFromCache(fileKey);
            if (cachedFile) {
                urlRef.value = cachedFile.url;
                return urlRef;
            }

            // 检查是否正在获取中，避免重复请求
            const fetchKey = `${fileKey}_${fileType}`;
            if (templateUrlFetching.value.has(fetchKey)) {
                console.log('createTemplateFileUrlRef: 正在获取中，跳过重复请求', { fileKey, fileType });
                return urlRef;
            }

            // 标记为正在获取
            templateUrlFetching.value.add(fetchKey);

            // 如果缓存中没有，异步获取
            getTemplateFileUrlFromApi(fileKey, fileType).then(url => {
                if (url) {
                    urlRef.value = url;
                    // 将获取到的URL存储到缓存中
                    setTemplateFileToCache(fileKey, { url, timestamp: Date.now() });
                }
            }).catch(error => {
                console.warn('获取模板文件URL失败:', error);
            }).finally(() => {
                // 移除获取状态
                templateUrlFetching.value.delete(fetchKey);
            });

            return urlRef;
        };

        // 创建响应式的任务文件URL（用于首屏渲染）
        const createTaskFileUrlRef = (taskId, fileKey) => {
            const urlRef = ref(null);

            // 检查参数有效性
            if (!taskId || !fileKey) {
                console.warn('createTaskFileUrlRef: 参数为空', { taskId, fileKey });
                return urlRef;
            }

            // 先从缓存获取
            const cachedFile = getTaskFileFromCache(taskId, fileKey);
            if (cachedFile) {
                urlRef.value = cachedFile.url;
                return urlRef;
            }

            // 如果缓存中没有，异步获取
            getTaskFileUrl(taskId, fileKey).then(url => {
                if (url) {
                    urlRef.value = url;
                    // 将获取到的URL存储到缓存中
                    setTaskFileToCache(taskId, fileKey, { url, timestamp: Date.now() });
                }
            }).catch(error => {
                console.warn('获取任务文件URL失败:', error);
            });

            return urlRef;
        };

        // 获取模板文件URL（异步版本，用于预加载等场景）
        const getTemplateFileUrlAsync = async (fileKey, fileType) => {
            // 检查参数有效性（静默处理，不打印警告，因为模板可能确实没有某些输入）
            if (!fileKey) {
                return null;
            }

            // 先从缓存获取
            const cachedFile = getTemplateFileFromCache(fileKey);
            if (cachedFile) {
                console.log('getTemplateFileUrlAsync: 从缓存获取', { fileKey, url: cachedFile.url });
                return cachedFile.url;
            }

            // 检查是否正在获取中，避免重复请求
            const fetchKey = `${fileKey}_${fileType}`;
            if (templateUrlFetching.value.has(fetchKey)) {
                console.log('getTemplateFileUrlAsync: 正在获取中，等待完成', { fileKey, fileType });
                // 等待其他请求完成
                return new Promise((resolve) => {
                    const checkInterval = setInterval(() => {
                        const cachedFile = getTemplateFileFromCache(fileKey);
                        if (cachedFile) {
                            clearInterval(checkInterval);
                            resolve(cachedFile.url);
                        } else if (!templateUrlFetching.value.has(fetchKey)) {
                            clearInterval(checkInterval);
                            resolve(null);
                        }
                    }, 100);
                });
            }

            // 标记为正在获取
            templateUrlFetching.value.add(fetchKey);

            // 如果缓存中没有，异步获取
            try {
                const url = await getTemplateFileUrlFromApi(fileKey, fileType);
                if (url) {
                    // 将获取到的URL存储到缓存中
                    setTemplateFileToCache(fileKey, { url, timestamp: Date.now() });
                }
                return url;
            } catch (error) {
                console.warn('getTemplateFileUrlAsync: 获取URL失败', error);
                return null;
            } finally {
                // 移除获取状态
                templateUrlFetching.value.delete(fetchKey);
            }
        };

        // 任务文件缓存管理函数
        const loadTaskFilesFromCache = () => {
            try {
                const cached = localStorage.getItem(TASK_FILE_CACHE_KEY);
                if (cached) {
                    const data = JSON.parse(cached);
                    // 检查是否过期
                    if (Date.now() - data.timestamp < TASK_FILE_CACHE_EXPIRY) {
                        // 将缓存数据加载到内存缓存中
                        for (const [cacheKey, fileData] of Object.entries(data.files)) {
                            taskFileCache.value.set(cacheKey, fileData);
                        }
                        return true;
                    } else {
                        // 缓存过期，清除
                        localStorage.removeItem(TASK_FILE_CACHE_KEY);
                    }
                }
            } catch (error) {
                console.warn('加载任务文件缓存失败:', error);
                localStorage.removeItem(TASK_FILE_CACHE_KEY);
            }
            return false;
        };

        const saveTaskFilesToCache = () => {
            try {
                const files = {};
                for (const [cacheKey, fileData] of taskFileCache.value.entries()) {
                    files[cacheKey] = fileData;
                }
                const data = {
                    files,
                    timestamp: Date.now()
                };
                localStorage.setItem(TASK_FILE_CACHE_KEY, JSON.stringify(data));
            } catch (error) {
                console.warn('保存任务文件缓存失败:', error);
            }
        };

        // 生成缓存键
        const getTaskFileCacheKey = (taskId, fileKey) => {
            return `${taskId}_${fileKey}`;
        };

        // 从缓存获取任务文件
        const getTaskFileFromCache = (taskId, fileKey) => {
            const cacheKey = getTaskFileCacheKey(taskId, fileKey);
            return taskFileCache.value.get(cacheKey) || null;
        };

        // 设置任务文件到缓存
        const setTaskFileToCache = (taskId, fileKey, fileData) => {
            const cacheKey = getTaskFileCacheKey(taskId, fileKey);
            taskFileCache.value.set(cacheKey, fileData);
            // 异步保存到localStorage
            setTimeout(() => {
                saveTaskFilesToCache();
            }, 100);
        };

        const getTaskFileUrlFromApi = async (taskId, fileKey, filename = null) => {
            let apiUrl = `/api/v1/task/input_url?task_id=${taskId}&name=${fileKey}`;
            if (filename) {
                apiUrl += `&filename=${encodeURIComponent(filename)}`;
            }
            if (fileKey.includes('output')) {
                apiUrl = `/api/v1/task/result_url?task_id=${taskId}&name=${fileKey}`;
            }
            const response = await apiRequest(apiUrl);
            if (response && response.ok) {
                const data = await response.json();
                let assertUrl = data.url;
                if (assertUrl.startsWith('./assets/')) {
                    const token = localStorage.getItem('accessToken');
                    if (token) {
                        assertUrl = `${assertUrl}&token=${encodeURIComponent(token)}`;
                    }
                }
                const cacheKey = filename ? `${fileKey}_${filename}` : fileKey;
                setTaskFileToCache(taskId, cacheKey, {
                    url: assertUrl,
                    timestamp: Date.now()
                });
                return assertUrl;
            } else if (response && response.status === 400) {
                // Handle directory input error (multi-person mode)
                try {
                    const errorData = await response.json();
                    if (errorData.error && errorData.error.includes('directory')) {
                        console.warn(`Input ${fileKey} is a directory (multi-person mode), cannot get single file URL`);
                        return null;
                    }
                } catch (e) {
                    // Ignore JSON parse errors
                }
            }
            return null;
        };

        // Podcast 音频 URL 缓存管理函数（模仿任务文件缓存）
        const loadPodcastAudioFromCache = () => {
            try {
                const cached = localStorage.getItem(PODCAST_AUDIO_CACHE_KEY);
                if (cached) {
                    const data = JSON.parse(cached);
                    // 检查是否过期
                    if (Date.now() - data.timestamp < PODCAST_AUDIO_CACHE_EXPIRY) {
                        // 将缓存数据加载到内存缓存中
                        for (const [cacheKey, audioData] of Object.entries(data.audio_urls)) {
                            podcastAudioCache.value.set(cacheKey, audioData);
                        }
                        podcastAudioCacheLoaded.value = true;
                        return true;
                    } else {
                        // 缓存过期，清除
                        localStorage.removeItem(PODCAST_AUDIO_CACHE_KEY);
                    }
                }
            } catch (error) {
                console.warn('加载播客音频缓存失败:', error);
                localStorage.removeItem(PODCAST_AUDIO_CACHE_KEY);
            }
            podcastAudioCacheLoaded.value = true;
            return false;
        };

        const savePodcastAudioToCache = () => {
            try {
                const audio_urls = {};
                for (const [cacheKey, audioData] of podcastAudioCache.value.entries()) {
                    audio_urls[cacheKey] = audioData;
                }
                const data = {
                    audio_urls,
                    timestamp: Date.now()
                };
                localStorage.setItem(PODCAST_AUDIO_CACHE_KEY, JSON.stringify(data));
            } catch (error) {
                console.warn('保存播客音频缓存失败:', error);
            }
        };

        // 生成播客音频缓存键
        const getPodcastAudioCacheKey = (sessionId) => {
            return sessionId;
        };

        // 从缓存获取播客音频 URL
        const getPodcastAudioFromCache = (sessionId) => {
            const cacheKey = getPodcastAudioCacheKey(sessionId);
            return podcastAudioCache.value.get(cacheKey) || null;
        };

        // 设置播客音频 URL 到缓存
        const setPodcastAudioToCache = (sessionId, audioData) => {
            const cacheKey = getPodcastAudioCacheKey(sessionId);
            podcastAudioCache.value.set(cacheKey, audioData);
            // 异步保存到localStorage
            setTimeout(() => {
                savePodcastAudioToCache();
            }, 100);
        };

        // 从 API 获取播客音频 URL（CDN URL）
        const getPodcastAudioUrlFromApi = async (sessionId) => {
            try {
                const response = await apiCall(`/api/v1/podcast/session/${sessionId}/audio_url`);
                if (response && response.ok) {
                    const data = await response.json();
                    const audioUrl = data.audio_url;
                    setPodcastAudioToCache(sessionId, {
                        url: audioUrl,
                        timestamp: Date.now()
                    });
                    return audioUrl;
                }
            } catch (error) {
                console.warn(`Failed to get audio URL for session ${sessionId}:`, error);
            }
            return null;
        };

        // 获取任务文件URL（优先从缓存，缓存没有则调用后端）
        const getTaskFileUrl = async (taskId, fileKey) => {
            // 先从缓存获取
            const cachedFile = getTaskFileFromCache(taskId, fileKey);
            if (cachedFile) {
                return cachedFile.url;
            }
            return await getTaskFileUrlFromApi(taskId, fileKey);
        };

        // 同步获取任务文件URL（仅从缓存获取，用于模板显示）
        const getTaskFileUrlSync = (taskId, fileKey) => {
            const cachedFile = getTaskFileFromCache(taskId, fileKey);
            if (cachedFile) {
                console.log('getTaskFileUrlSync: 从缓存获取', { taskId, fileKey, url: cachedFile.url, type: typeof cachedFile.url });
                return cachedFile.url;
            }
            console.log('getTaskFileUrlSync: 缓存中没有找到', { taskId, fileKey });
            return null;
        };

        // 预加载任务文件
        const preloadTaskFilesUrl = async (tasks) => {
            if (!tasks || tasks.length === 0) return;

            // 先尝试从localStorage加载缓存
            if (taskFileCache.value.size === 0) {
                loadTaskFilesFromCache();
            }

            console.log(`开始获取 ${tasks.length} 个任务的文件url`);

            // 分批预加载，避免过多并发请求
            const batchSize = 5;
            for (let i = 0; i < tasks.length; i += batchSize) {
                const batch = tasks.slice(i, i + batchSize);

                const promises = batch.map(async (task) => {
                    if (!task.task_id) return;

                    // 预加载输入图片
                    if (task.inputs && task.inputs.input_image) {
                        await getTaskFileUrl(task.task_id, 'input_image');
                    }
                    // 预加载输入音频
                    if (task.inputs && task.inputs.input_audio) {
                        await getTaskFileUrl(task.task_id, 'input_audio');
                    }
                    // 预加载输出视频
                    if (task.outputs && task.outputs.output_video && task.status === 'SUCCEED') {
                        await getTaskFileUrl(task.task_id, 'output_video');
                    }
                });

                await Promise.all(promises);

                // 批次间添加延迟
                if (i + batchSize < tasks.length) {
                    await new Promise(resolve => setTimeout(resolve, 200));
                }
            }

            console.log('任务文件url预加载完成');
        };

        // 预加载模板文件
        const preloadTemplateFilesUrl = async (templates) => {
            if (!templates || templates.length === 0) return;

            // 先尝试从localStorage加载缓存
            if (templateFileCache.value.size === 0) {
                loadTemplateFilesFromCache();
            }

            console.log(`开始获取 ${templates.length} 个模板的文件url`);

            // 分批预加载，避免过多并发请求
            const batchSize = 5;
            for (let i = 0; i < templates.length; i += batchSize) {
                const batch = templates.slice(i, i + batchSize);

                const promises = batch.map(async (template) => {
                    if (!template.task_id) return;

                    // 预加载视频文件
                    if (template.outputs?.output_video) {
                        await getTemplateFileUrlAsync(template.outputs.output_video, 'videos');
                    }

                    // 预加载图片文件
                    if (template.inputs?.input_image) {
                        await getTemplateFileUrlAsync(template.inputs.input_image, 'images');
                    }

                    // 预加载音频文件
                    if (template.inputs?.input_audio) {
                        await getTemplateFileUrlAsync(template.inputs.input_audio, 'audios');
                    }
                });

                await Promise.all(promises);

                // 批次间添加延迟
                if (i + batchSize < templates.length) {
                    await new Promise(resolve => setTimeout(resolve, 200));
                }
            }

            console.log('模板文件url预加载完成');
        };

        const refreshTasks = async (forceRefresh = false) => {
            try {
                console.log('开始刷新任务列表, forceRefresh:', forceRefresh, 'currentPage:', currentTaskPage.value);

                // 构建缓存键，包含分页和过滤条件
                const cacheKey = `${TASKS_CACHE_KEY}_${currentTaskPage.value}_${taskPageSize.value}_${statusFilter.value}_${taskSearchQuery.value}`;

                // 如果不是强制刷新，先尝试从缓存加载
                if (!forceRefresh) {
                    const cachedTasks = loadFromCache(cacheKey, TASKS_CACHE_EXPIRY);
                    if (cachedTasks) {
                        console.log('从缓存加载任务列表');
                        tasks.value = cachedTasks.tasks || [];
                        pagination.value = cachedTasks.pagination || null;
                        // 强制触发响应式更新
                        await nextTick();
                        // 强制刷新分页组件
                        paginationKey.value++;
                        // 使用新的任务文件预加载逻辑
                        await preloadTaskFilesUrl(tasks.value);
                        return;
                    }
                }

                const params = new URLSearchParams({
                    page: currentTaskPage.value.toString(),
                    page_size: taskPageSize.value.toString()
                });

                if (statusFilter.value !== 'ALL') {
                    params.append('status', statusFilter.value);
                }

                console.log('请求任务列表API:', `/api/v1/task/list?${params.toString()}`);
                const response = await apiRequest(`/api/v1/task/list?${params.toString()}`);
                if (response && response.ok) {
                    const data = await response.json();
                    console.log('任务列表API响应:', data);

                    // 强制清空并重新赋值，确保Vue检测到变化
                    tasks.value = [];
                    pagination.value = null;
                    await nextTick();

                    tasks.value = data.tasks || [];
                    pagination.value = data.pagination || null;

                    // 缓存任务数据
                    saveToCache(cacheKey, {
                        tasks: data.tasks || [],
                        pagination: data.pagination || null
                    });
                    console.log('缓存任务列表数据成功');

                    // 强制触发响应式更新
                    await nextTick();

                    // 强制刷新分页组件
                    paginationKey.value++;

                    // 使用新的任务文件预加载逻辑
                    await preloadTaskFilesUrl(tasks.value);
                } else if (response) {
                    showAlert(t('refreshTaskListFailed'), 'danger');
                }
                // 如果response为null，说明是认证错误，apiRequest已经处理了
            } catch (error) {
                console.error('刷新任务列表失败:', error);
                // showAlert(`刷新任务列表失败: ${error.message}`, 'danger');
            }
        };

        // 分页相关函数
        const goToPage = async (page) => {
            isPageLoading.value = true;
            if (page < 1 || page > pagination.value?.total_pages || page === currentTaskPage.value) {
                isPageLoading.value = false;
                return;
            }
            currentTaskPage.value = page;
            taskPageInput.value = page; // 同步更新输入框
            await refreshTasks();
            isPageLoading.value = false;
        };

        const jumpToPage = async () => {
            const page = parseInt(taskPageInput.value);
            if (page && page >= 1 && page <= pagination.value?.total_pages && page !== currentTaskPage.value) {
                await goToPage(page);
            } else {
                // 如果输入无效，恢复到当前页
                taskPageInput.value = currentTaskPage.value;
            }
        };

        // Template分页相关函数
        const goToTemplatePage = async (page) => {
            isPageLoading.value=true;
            if (page < 1 || page > templatePagination.value?.total_pages || page === templateCurrentPage.value) {
                isPageLoading.value = false;
                return;
            }
            templateCurrentPage.value = page;
            templatePageInput.value = page; // 同步更新输入框
            await loadImageAudioTemplates();
            isPageLoading.value = false;
        };

        const jumpToTemplatePage = async () => {
            const page = parseInt(templatePageInput.value);
            if (page && page >= 1 && page <= templatePagination.value?.total_pages && page !== templateCurrentPage.value) {
                await goToTemplatePage(page);
            } else {
                // 如果输入无效，恢复到当前页
                templatePageInput.value = templateCurrentPage.value;
            }
        };

        const getVisiblePages = () => {
            if (!pagination.value) return [];

            const totalPages = pagination.value.total_pages;
            const current = currentTaskPage.value;
            const pages = [];

            // 总是显示第一页
            pages.push(1);

            if (totalPages <= 5) {
                // 如果总页数少于等于7页，显示所有页码
                for (let i = 2; i <= totalPages - 1; i++) {
                    pages.push(i);
                }
            } else {
                // 如果总页数大于7页，使用省略号
                if (current <= 3) {
                    // 当前页在前4页
                    for (let i = 2; i <= 3; i++) {
                        pages.push(i);
                    }
                    pages.push('...');
                } else if (current >= totalPages - 2) {
                    // 当前页在后4页
                    pages.push('...');
                    for (let i = totalPages - 2; i <= totalPages - 1; i++) {
                        pages.push(i);
                    }
                } else {
                    // 当前页在中间
                    pages.push('...');
                    for (let i = current - 1; i <= current + 1; i++) {
                        pages.push(i);
                    }
                    pages.push('...');
                }
            }

            // 总是显示最后一页（如果不是第一页）
            if (totalPages > 1) {
                pages.push(totalPages);
            }

            return pages;
        };

        const getVisibleTemplatePages = () => {
            if (!templatePagination.value) return [];

            const totalPages = templatePagination.value.total_pages;
            const current = templateCurrentPage.value;
            const pages = [];

            // 总是显示第一页
            pages.push(1);

            if (totalPages <= 5) {
                // 如果总页数少于等于7页，显示所有页码
                for (let i = 2; i <= totalPages - 1; i++) {
                    pages.push(i);
                }
            } else {
                // 显示当前页附近的页码
                const start = Math.max(2, current - 1);
                const end = Math.min(totalPages - 1, current + 1);

                if (start > 2) {
                    pages.push('...');
                }

                for (let i = start; i <= end; i++) {
                    if (i !== 1 && i !== totalPages) {
                        pages.push(i);
                    }
                }

                if (end < totalPages - 1) {
                    pages.push('...');
                }
            }

            // 总是显示最后一页
            if (totalPages > 1) {
                pages.push(totalPages);
            }

            return pages;
        };

        // 灵感广场分页相关函数
        const goToInspirationPage = async (page) => {
            isPageLoading.value = true;
            if (page < 1 || page > inspirationPagination.value?.total_pages || page === inspirationCurrentPage.value) {
                isPageLoading.value = false;
                return;
            }
            inspirationCurrentPage.value = page;
            inspirationPageInput.value = page; // 同步更新输入框
            await loadInspirationData();
            isPageLoading.value = false;
        };

        const jumpToInspirationPage = async () => {
            const page = parseInt(inspirationPageInput.value);
            if (page && page >= 1 && page <= inspirationPagination.value?.total_pages && page !== inspirationCurrentPage.value) {
                await goToInspirationPage(page);
            } else {
                // 如果输入无效，恢复到当前页
                inspirationPageInput.value = inspirationCurrentPage.value;
            }
        };

        const getVisibleInspirationPages = () => {
            if (!inspirationPagination.value) return [];

            const totalPages = inspirationPagination.value.total_pages;
            const current = inspirationCurrentPage.value;
            const pages = [];

            // 总是显示第一页
            pages.push(1);

            if (totalPages <= 5) {
                // 如果总页数少于等于7页，显示所有页码
                for (let i = 2; i <= totalPages - 1; i++) {
                    pages.push(i);
                }
            } else {
                // 显示当前页附近的页码
                const start = Math.max(2, current - 1);
                const end = Math.min(totalPages - 1, current + 1);

                if (start > 2) {
                    pages.push('...');
                }

                for (let i = start; i <= end; i++) {
                    if (i !== 1 && i !== totalPages) {
                        pages.push(i);
                    }
                }

                if (end < totalPages - 1) {
                    pages.push('...');
                }
            }

            // 总是显示最后一页
            if (totalPages > 1) {
                pages.push(totalPages);
            }

            return pages;
        };

        const getStatusBadgeClass = (status) => {
            const statusMap = {
                'SUCCEED': 'bg-success',
                'FAILED': 'bg-danger',
                'RUNNING': 'bg-warning',
                'PENDING': 'bg-secondary',
                'CREATED': 'bg-secondary'
            };
            return statusMap[status] || 'bg-secondary';
        };

        const viewSingleResult = async (taskId, key) => {
            try {
                downloadLoading.value = true;
                const url = await getTaskFileUrl(taskId, key);
                if (url) {
                    const response = await fetch(url);
                    if (response.ok) {
                        const blob = await response.blob();
                        const videoBlob = new Blob([blob], { type: 'video/mp4' });
                        const url = window.URL.createObjectURL(videoBlob);
                        window.open(url, '_blank');
                    } else {
                        showAlert(t('getResultFailed'), 'danger');
                    }
                } else {
                    showAlert(t('getTaskResultFailedAlert'), 'danger');
                }
            } catch (error) {
                showAlert(`${t('viewTaskResultFailedAlert')}: ${error.message}`, 'danger');
            } finally {
                downloadLoading.value = false;
            }
        };

        const cancelTask = async (taskId, fromDetailPage = false) => {
            try {
                // 显示确认对话框
                const confirmed = await showConfirmDialog({
                    title: t('cancelTaskConfirm'),
                    message: t('cancelTaskConfirmMessage'),
                    confirmText: t('confirmCancel'),
                });

                if (!confirmed) {
                    return;
                }

                const response = await apiRequest(`/api/v1/task/cancel?task_id=${taskId}`);
                if (response && response.ok) {
                    showAlert(t('taskCancelSuccessAlert'), 'success');

                    // 如果当前在任务详情界面，刷新任务后关闭详情弹窗
                    if (fromDetailPage) {
                        refreshTasks(true); // 强制刷新
                        const updatedTask = tasks.value.find(t => t.task_id === taskId);
                        if (updatedTask) {
                            modalTask.value = updatedTask;
                        }
                        await nextTick();
                        closeTaskDetailModal();
                    } else {
                        refreshTasks(true); // 强制刷新
                    }

                } else if (response) {
                    const error = await response.json();
                    showAlert(`${t('cancelTaskFailedAlert')}: ${error.message}`, 'danger');
                }
                // 如果response为null，说明是认证错误，apiRequest已经处理了
            } catch (error) {
                showAlert(`${t('cancelTaskFailedAlert')}: ${error.message}`, 'danger');
            }
        };

        const resumeTask = async (taskId, fromDetailPage = false) => {
            try {
                // 先获取任务信息，检查任务状态
                const taskResponse = await apiRequest(`/api/v1/task/query?task_id=${taskId}`);
                if (!taskResponse || !taskResponse.ok) {
                    showAlert(t('taskNotFoundAlert'), 'danger');
                    return;
                }

                const task = await taskResponse.json();

                // 如果任务已完成，则删除并重新生成
                if (task.status === 'SUCCEED') {
                    // 显示确认对话框
                    const confirmed = await showConfirmDialog({
                        title: t('regenerateTaskConfirm'),
                        message: t('regenerateTaskConfirmMessage'),
                        confirmText: t('confirmRegenerate')
                    });

                    if (!confirmed) {
                        return;
                    }

                    // 显示重新生成中的提示
                    showAlert(t('regeneratingTaskAlert'), 'info');

                    const deleteResponse = await apiRequest(`/api/v1/task/delete?task_id=${taskId}`, {
                        method: 'DELETE'
                    });
                    if (!deleteResponse || !deleteResponse.ok) {
                        showAlert(t('deleteTaskFailedAlert'), 'danger');
                        return;
                    }
                    try {
                        // 设置任务类型
                        selectedTaskId.value = task.task_type;
                        console.log('selectedTaskId.value', selectedTaskId.value);

                        // 获取当前表单
                        const currentForm = getCurrentForm();

                        // 设置模型
                        if (task.params && task.params.model_cls) {
                            currentForm.model_cls = task.params.model_cls;
                        }

                        // 设置prompt
                        if (task.params && task.params.prompt) {
                            currentForm.prompt = task.params.prompt;
                        }

                        // localStorage 不再保存文件内容，直接从后端获取任务文件
                            try {
                                // 使用现有的函数获取图片和音频URL
                                const imageUrl = await getTaskInputImage(task);
                                const audioUrl = await getTaskInputAudio(task);

                                // 加载图片文件
                                if (imageUrl) {
                                    try {
                                        const imageResponse = await fetch(imageUrl);
                                        if (imageResponse && imageResponse.ok) {
                                            const blob = await imageResponse.blob();
                                            const filename = task.inputs[Object.keys(task.inputs).find(key =>
                                                key.includes('image') ||
                                                task.inputs[key].toString().toLowerCase().match(/\.(jpg|jpeg|png|gif|bmp|webp)$/)
                                            )] || 'image.jpg';
                                            const file = new File([blob], filename, { type: blob.type });
                                            currentForm.imageFile = file;
                                            setCurrentImagePreview(URL.createObjectURL(file));
                                        }
                                    } catch (error) {
                                        console.warn('Failed to load image file:', error);
                                    }
                                }

                                // 加载音频文件
                                if (audioUrl) {
                                    try {
                                        const audioResponse = await fetch(audioUrl);
                                        if (audioResponse && audioResponse.ok) {
                                            const blob = await audioResponse.blob();
                                            const filename = task.inputs[Object.keys(task.inputs).find(key =>
                                                key.includes('audio') ||
                                                task.inputs[key].toString().toLowerCase().match(/\.(mp3|wav|mp4|aac|ogg|m4a)$/)
                                            )] || 'audio.wav';

                                            // 根据文件扩展名确定正确的MIME类型
                                            let mimeType = blob.type;
                                            if (!mimeType || mimeType === 'application/octet-stream') {
                                                const ext = filename.toLowerCase().split('.').pop();
                                                const mimeTypes = {
                                                    'mp3': 'audio/mpeg',
                                                    'wav': 'audio/wav',
                                                    'mp4': 'audio/mp4',
                                                    'aac': 'audio/aac',
                                                    'ogg': 'audio/ogg',
                                                    'm4a': 'audio/mp4'
                                                };
                                                mimeType = mimeTypes[ext] || 'audio/mpeg';
                                            }

                                            const file = new File([blob], filename, { type: mimeType });
                                            currentForm.audioFile = file;
                                            console.log('复用任务 - 从后端加载音频文件:', {
                                                name: file.name,
                                                type: file.type,
                                                size: file.size,
                                                originalBlobType: blob.type
                                            });
                                            // 使用FileReader生成data URL，与正常上传保持一致
                                            const reader = new FileReader();
                                            reader.onload = (e) => {
                                                setCurrentAudioPreview(e.target.result);
                                                console.log('复用任务 - 音频预览已设置:', e.target.result.substring(0, 50) + '...');
                                            };
                                            reader.readAsDataURL(file);
                                        }

                                    } catch (error) {
                                        console.warn('Failed to load audio file:', error);
                                    }
                                }
                            } catch (error) {
                                console.warn('Failed to load task data from backend:', error);
                        }

                        showAlert(t('taskMaterialReuseSuccessAlert'), 'success');

                    } catch (error) {
                        console.error('Failed to resume task:', error);
                        showAlert(t('loadTaskDataFailedAlert'), 'danger');
                        return;
                    }
                    // 如果从详情页调用，关闭详情页
                    if (fromDetailPage) {
                        closeTaskDetailModal();
                    }

                    submitTask();


                    return; // 不需要继续执行后续的API调用
                } else {
                    // 对于未完成的任务，使用原有的恢复逻辑
                    const response = await apiRequest(`/api/v1/task/resume?task_id=${taskId}`);
                    if (response && response.ok) {
                        showAlert(t('taskRetrySuccessAlert'), 'success');

                        // 如果当前在任务详情界面，先刷新任务列表，然后重新获取任务信息
                        if (fromDetailPage) {
                            refreshTasks(true); // 强制刷新
                            const updatedTask = tasks.value.find(t => t.task_id === taskId);
                            if (updatedTask) {
                                selectedTask.value = updatedTask;
                            }
                            startPollingTask(taskId);
                            await nextTick();
                        } else {
                            refreshTasks(true); // 强制刷新

                            // 开始轮询新提交的任务状态
                            startPollingTask(taskId);
                        }
                    } else if (response) {
                        const error = await response.json();
                        showAlert(`${t('retryTaskFailedAlert')}: ${error.message}`, 'danger');
                    }
                }
            } catch (error) {
                console.error('resumeTask error:', error);
                showAlert(`${t('retryTaskFailedAlert')}: ${error.message}`, 'danger');
            }
        };

        // 切换任务菜单显示状态
        const toggleTaskMenu = (taskId) => {
            // 先关闭所有其他菜单
            closeAllTaskMenus();
            // 然后打开当前菜单
            taskMenuVisible.value[taskId] = true;
        };

        // 关闭所有任务菜单
        const closeAllTaskMenus = () => {
            taskMenuVisible.value = {};
        };

        // 点击外部关闭菜单
        const handleClickOutside = (event) => {
            if (!event.target.closest('.task-menu-container')) {
                closeAllTaskMenus();
            }
            if (!event.target.closest('.task-type-dropdown')) {
                showTaskTypeMenu.value = false;
            }
            if (!event.target.closest('.model-dropdown')) {
                showModelMenu.value = false;
            }
        };

        const deleteTask = async (taskId, fromDetailPage = false) => {
            try {
                // 显示确认对话框
                const confirmed = await showConfirmDialog({
                    title: t('deleteTaskConfirm'),
                    message: t('deleteTaskConfirmMessage'),
                    confirmText: t('confirmDelete')
                });

                if (!confirmed) {
                    return;
                }
                const response = await apiRequest(`/api/v1/task/delete?task_id=${taskId}`, {
                    method: 'DELETE'
                });

                if (response && response.ok) {
                    showAlert(t('taskDeletedSuccessAlert'), 'success');
                    const deletedTaskIndex = tasks.value.findIndex(task => task.task_id === taskId);
                    if (deletedTaskIndex !== -1) {
                        const wasCurrent = currentTask.value?.task_id === taskId;
                        tasks.value.splice(deletedTaskIndex, 1);
                        if (wasCurrent) {
                            currentTask.value = tasks.value[deletedTaskIndex] || tasks.value[deletedTaskIndex - 1] || null;
                        }
                    }
                    refreshTasks(true); // 强制刷新

                    // 如果是从任务详情页删除，删除成功后关闭详情弹窗
                    if (fromDetailPage) {
                        closeTaskDetailModal();
                        if (!selectedTaskId.value) {
                            if (availableTaskTypes.value.includes('s2v')) {
                                selectTask('s2v');
                            }
                        }
                    }
                } else if (response) {
                    const error = await response.json();
                    showAlert(`${t('deleteTaskFailedAlert')}: ${error.message}`, 'danger');
                }
                // 如果response为null，说明是认证错误，apiRequest已经处理了
            } catch (error) {
                showAlert(`${t('deleteTaskFailedAlert')}: ${error.message}`, 'danger');
            }
        };

        const loadTaskFiles = async (task) => {
            try {
                loadingTaskFiles.value = true;

                const files = { inputs: {}, outputs: {} };

                // 获取输入文件（所有状态的任务都需要）
                if (task.inputs) {
                    for (const [key, inputPath] of Object.entries(task.inputs)) {
                        try {
                            const url = await getTaskFileUrl(taskId, key);
                            if (url) {
                                const response = await fetch(url);
                                if (response && response.ok) {
                                    const blob = await response.blob()
                                    files.inputs[key] = {
                                        name: inputPath, // 使用原始文件名而不是key
                                        path: inputPath,
                                        blob: blob,
                                        url: URL.createObjectURL(blob)
                                    }
                                }
                            }
                        } catch (error) {
                            console.error(`Failed to load input ${key}:`, error);
                            files.inputs[key] = {
                                name: inputPath, // 使用原始文件名而不是key
                                path: inputPath,
                                error: true
                            };
                        }
                    }
                }

                // 只对成功完成的任务获取输出文件
                if (task.status === 'SUCCEED' && task.outputs) {
                    for (const [key, outputPath] of Object.entries(task.outputs)) {
                        try {
                            const url = await getTaskFileUrl(taskId, key);
                            if (url) {
                                const response = await fetch(url);
                                if (response && response.ok) {
                                    const blob = await response.blob()
                                    files.outputs[key] = {
                                        name: outputPath, // 使用原始文件名而不是key
                                        path: outputPath,
                                        blob: blob,
                                        url: URL.createObjectURL(blob)
                                    }
                                };
                            }
                        } catch (error) {
                            console.error(`Failed to load output ${key}:`, error);
                            files.outputs[key] = {
                                name: outputPath, // 使用原始文件名而不是key
                                path: outputPath,
                                error: true
                            };
                        }
                    }
                }

                selectedTaskFiles.value = files;

            } catch (error) {
                console.error('Failed to load task files: task_id=', taskId, error);
                showAlert(t('loadTaskFilesFailedAlert'), 'danger');
            } finally {
                loadingTaskFiles.value = false;
            }
        };

        const reuseTask = async (task) => {
            if (!task) {
                showAlert(t('loadTaskDataFailedAlert'), 'danger');
                return;
            }

            try {
                templateLoading.value = true;
                templateLoadingMessage.value = t('prefillLoadingTask');
                // 跳转到任务创建界面
                isCreationAreaExpanded.value = true;
                if (showTaskDetailModal.value) {
                    closeTaskDetailModal();
                }

                // 设置任务类型
                selectedTaskId.value = task.task_type;
                console.log('selectedTaskId.value', selectedTaskId.value);

                // 获取当前表单
                const currentForm = getCurrentForm();

                // 立即切换到创建视图，后续资产异步加载
                switchToCreateView();

                // 设置模型
                if (task.params && task.params.model_cls) {
                    currentForm.model_cls = task.params.model_cls;
                }

                // 设置prompt
                if (task.params && task.params.prompt) {
                    currentForm.prompt = task.params.prompt;
                }

                // localStorage 不再保存文件内容，直接从后端获取任务文件
                    try {
                        // 使用现有的函数获取图片和音频URL
                        const imageUrl = await getTaskInputImage(task);
                        const audioUrl = await getTaskInputAudio(task);


                        // 加载音频文件
                        if (audioUrl) {
                            try {
                                const audioResponse = await fetch(audioUrl);
                                if (audioResponse && audioResponse.ok) {
                                    // Check if the response is an error (for directory inputs)
                                    const contentType = audioResponse.headers.get('content-type');
                                    if (contentType && contentType.includes('application/json')) {
                                        const errorData = await audioResponse.json();
                                            // Not a directory error, proceed with normal loading
                                            currentForm.audioUrl = audioUrl;
                                            setCurrentAudioPreview(audioUrl);

                                            const blob = await audioResponse.blob();
                                            const filename = task.inputs[Object.keys(task.inputs).find(key =>
                                                key.includes('audio') ||
                                                task.inputs[key].toString().toLowerCase().match(/\.(mp3|wav|mp4|aac|ogg|m4a)$/)
                                            )] || 'audio.wav';

                                            // 根据文件扩展名确定正确的MIME类型
                                            let mimeType = blob.type;
                                            if (!mimeType || mimeType === 'application/octet-stream') {
                                                const ext = filename.toLowerCase().split('.').pop();
                                                const mimeTypes = {
                                                    'mp3': 'audio/mpeg',
                                                    'wav': 'audio/wav',
                                                    'mp4': 'audio/mp4',
                                                    'aac': 'audio/aac',
                                                    'ogg': 'audio/ogg',
                                                    'm4a': 'audio/mp4'
                                                };
                                                mimeType = mimeTypes[ext] || 'audio/mpeg';
                                            }

                                            const file = new File([blob], filename, { type: mimeType });
                                            currentForm.audioFile = file;
                                            console.log('复用任务 - 从后端加载音频文件:', {
                                                name: file.name,
                                                type: file.type,
                                                size: file.size,
                                                originalBlobType: blob.type
                                            });
                                    } else {
                                        // Normal audio file response
                                        currentForm.audioUrl = audioUrl;
                                        setCurrentAudioPreview(audioUrl);

                                        const blob = await audioResponse.blob();
                                        const filename = task.inputs[Object.keys(task.inputs).find(key =>
                                            key.includes('audio') ||
                                            task.inputs[key].toString().toLowerCase().match(/\.(mp3|wav|mp4|aac|ogg|m4a)$/)
                                        )] || 'audio.wav';

                                        // 根据文件扩展名确定正确的MIME类型
                                        let mimeType = blob.type;
                                        if (!mimeType || mimeType === 'application/octet-stream') {
                                            const ext = filename.toLowerCase().split('.').pop();
                                            const mimeTypes = {
                                                'mp3': 'audio/mpeg',
                                                'wav': 'audio/wav',
                                                'mp4': 'audio/mp4',
                                                'aac': 'audio/aac',
                                                'ogg': 'audio/ogg',
                                                'm4a': 'audio/mp4'
                                            };
                                            mimeType = mimeTypes[ext] || 'audio/mpeg';
                                        }

                                        const file = new File([blob], filename, { type: mimeType });
                                        currentForm.audioFile = file;
                                        console.log('复用任务 - 从后端加载音频文件:', {
                                            name: file.name,
                                            type: file.type,
                                            size: file.size,
                                            originalBlobType: blob.type
                                        });
                                    }
                                }
                            } catch (error) {
                                console.warn('Failed to load audio file:', error);
                            }
                        }

                                                // 加载图片文件
                                                if (imageUrl) {
                                                    try {
                                                        const imageResponse = await fetch(imageUrl);
                                                        if (imageResponse && imageResponse.ok) {
                                                            const blob = await imageResponse.blob();
                                                            const filename = task.inputs[Object.keys(task.inputs).find(key =>
                                                                key.includes('image') ||
                                                                task.inputs[key].toString().toLowerCase().match(/\.(jpg|jpeg|png|gif|bmp|webp)$/)
                                                            )] || 'image.jpg';
                                                            const file = new File([blob], filename, { type: blob.type });
                                                            currentForm.imageFile = file;
                                                            const imagePreviewUrl = URL.createObjectURL(file);
                                                            setCurrentImagePreview(imageUrl);

                                                            // Reset detected faces
                                                            if (selectedTaskId.value === 'i2v') {
                                                                i2vForm.value.detectedFaces = [];
                                                            } else if (selectedTaskId.value === 's2v') {
                                                                s2vForm.value.detectedFaces = [];
                                                            }

                                                            // 不再自动检测人脸，等待用户手动打开多角色模式开关
                                                        }
                                                    } catch (error) {
                                                        console.warn('Failed to load image file:', error);
                                                    }
                                                }
                    } catch (error) {
                        console.warn('Failed to load task data from backend:', error);
                }

                showAlert(t('taskMaterialReuseSuccessAlert'), 'success');

            } catch (error) {
                console.error('Failed to reuse task:', error);
                showAlert(t('loadTaskDataFailedAlert'), 'danger');
            } finally {
                templateLoading.value = false;
                templateLoadingMessage.value = '';
            }
        };

        const downloadFile = async (fileInfo) => {
            if (!fileInfo || !fileInfo.blob) {
                showAlert(t('fileUnavailableAlert'), 'danger');
                return false;
            }

            const blob = fileInfo.blob;
            const fileName = fileInfo.name || 'download';
            const mimeType = blob.type || fileInfo.mimeType || 'application/octet-stream';

            try {
                const objectUrl = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = objectUrl;
                a.download = fileName;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(objectUrl);
                showAlert(t('downloadSuccessAlert'), 'success');
                return true;
            } catch (error) {
                console.error('Download failed:', error);
                showAlert(t('downloadFailedAlert'), 'danger');
                return false;
            }
        };

        // 处理文件下载
        const handleDownloadFile = async (taskId, fileKey, fileName) => {
            if (downloadLoading.value) {
                showAlert(t('downloadInProgressNotice'), 'info');
                return;
            }

            downloadLoading.value = true;
            downloadLoadingMessage.value = t('downloadPreparing');

            try {
                console.log('开始下载文件:', { taskId, fileKey, fileName });

                // 处理文件名，确保有正确的后缀名
                let finalFileName = fileName;
                if (fileName && typeof fileName === 'string') {
                    const hasExtension = /\.[a-zA-Z0-9]+$/.test(fileName);
                    if (!hasExtension) {
                        const extension = getFileExtension(fileKey);
                        finalFileName = `${fileName}.${extension}`;
                        console.log('添加后缀名:', finalFileName);
                    }
                } else {
                    finalFileName = `${fileKey}.${getFileExtension(fileKey)}`;
                }

                downloadLoadingMessage.value = t('downloadFetching');

                let downloadUrl = null;

                const cachedData = getTaskFileFromCache(taskId, fileKey);
                if (cachedData?.url) {
                    downloadUrl = cachedData.url;
                }

                if (!downloadUrl) {
                    downloadUrl = await getTaskFileUrl(taskId, fileKey);
                }

                if (!downloadUrl) {
                    throw new Error('无法获取文件URL');
                }

                const response = await fetch(downloadUrl);
                if (!response.ok) {
                    throw new Error(`文件响应失败: ${response.status}`);
                }

                const blob = await response.blob();
                downloadLoadingMessage.value = t('downloadSaving');
                await downloadFile({
                    blob,
                    name: finalFileName,
                    mimeType: blob.type
                });
            } catch (error) {
                console.error('下载失败:', error);
                showAlert(t('downloadFailedAlert'), 'danger');
            } finally {
                downloadLoading.value = false;
                downloadLoadingMessage.value = '';
            }
        }

        const viewFile = (fileInfo) => {
            if (!fileInfo || !fileInfo.url) {
                showAlert(t('fileUnavailableAlert'), 'danger');
                return;
            }

            // 在新窗口中打开文件
            window.open(fileInfo.url, '_blank');
        };

        const clearTaskFiles = () => {
            // 清理 URL 对象，释放内存
            Object.values(selectedTaskFiles.value.inputs).forEach(file => {
                if (file.url) {
                    URL.revokeObjectURL(file.url);
                }
            });
            Object.values(selectedTaskFiles.value.outputs).forEach(file => {
                if (file.url) {
                    URL.revokeObjectURL(file.url);
                }
            });
            selectedTaskFiles.value = { inputs: {}, outputs: {} };
        };

        const showTaskCreator = () => {
            selectedTask.value = null;
            // clearTaskFiles(); // 清空文件缓存
            selectedTaskId.value = 's2v'; // 默认选择数字人任务

            // 停止所有任务状态轮询
            pollingTasks.value.clear();
            if (pollingInterval.value) {
                clearInterval(pollingInterval.value);
                pollingInterval.value = null;
            }
        };

        const toggleSidebar = () => {
            sidebarCollapsed.value = !sidebarCollapsed.value;

            if (sidebarCollapsed.value) {
                // 收起时，将历史任务栏隐藏到屏幕左侧
                if (sidebar.value) {
                    sidebar.value.style.transform = 'translateX(-100%)';
                }
            } else {
                // 展开时，恢复历史任务栏位置
                if (sidebar.value) {
                    sidebar.value.style.transform = 'translateX(0)';
                }
            }

            // 更新悬浮按钮位置
            updateFloatingButtonPosition(sidebarWidth.value);
        };

        const clearPrompt = () => {
            getCurrentForm().prompt = '';
            updateUploadedContentStatus();
        };

        const getTaskItemClass = (status) => {
            if (status === 'SUCCEED') return 'bg-laser-purple/15 border border-laser-purple/30';
            if (status === 'RUNNING') return 'bg-laser-purple/15 border border-laser-purple/30';
            if (status === 'FAILED') return 'bg-red-500/15 border border-red-500/30';
            return 'bg-dark-light border border-gray-700';
        };

        const getStatusIndicatorClass = (status) => {
        const base = 'inline-block w-2 aspect-square rounded-full shrink-0 align-middle';
            if (status === 'SUCCEED')
                return `${base} bg-gradient-to-r from-emerald-200 to-green-300 shadow-md shadow-emerald-300/30`;
            if (status === 'RUNNING')
                return `${base} bg-gradient-to-r from-amber-200 to-yellow-300 shadow-md shadow-amber-300/30 animate-pulse`;
            if (status === 'FAILED')
                return `${base} bg-gradient-to-r from-red-200 to-pink-300 shadow-md shadow-red-300/30`;
            return `${base} bg-gradient-to-r from-gray-200 to-gray-300 shadow-md shadow-gray-300/30`;
            };

        const getTaskTypeBtnClass = (taskType) => {
            if (selectedTaskId.value === taskType) {
                return 'text-gradient-icon border-b-2 border-laser-purple';
            }
            return 'text-gray-400 hover:text-gradient-icon';
        };

        const getModelBtnClass = (model) => {
            if (getCurrentForm().model_cls === model) {
                return 'bg-laser-purple/20 border border-laser-purple/40 active shadow-laser';
            }
            return 'bg-dark-light border border-gray-700 hover:bg-laser-purple/15 hover:border-laser-purple/40 transition-all hover:shadow-laser';
        };

        const getTaskTypeIcon = (taskType) => {
            const iconMap = {
                't2v': 'fas fa-font',  // 文字A形图标
                'i2v': 'fas fa-image',     // 图像图标
                's2v': 'fas fa-user', // 人物图标
                'animate': 'fi fi-br-running text-lg' // 角色替换图标
            };
            return iconMap[taskType] || 'fas fa-video';
        };

        const getTaskTypeName = (task) => {
            // 如果传入的是字符串，直接返回映射
            if (!task) {
                return '未知';
            }
            if (typeof task === 'string') {
                return nameMap.value[task] || task;
            }

            // 如果传入的是任务对象，根据模型类型判断
            if (task && task.model_cls) {
                const modelCls = task.model_cls.toLowerCase();

                return nameMap.value[task.task_type] || task.task_type;
            }

            // 默认返回task_type
            return task.task_type || '未知';
        };

        const getPromptPlaceholder = () => {
            if (selectedTaskId.value === 't2v') {
                return t('pleaseEnterThePromptForVideoGeneration') + '，'+ t('describeTheContentStyleSceneOfTheVideo');
            } else if (selectedTaskId.value === 'i2v') {
                return t('pleaseEnterThePromptForVideoGeneration') + '，'+ t('describeTheContentActionRequirementsBasedOnTheImage');
            } else if (selectedTaskId.value === 's2v') {
                return t('optional') + ' '+ t('pleaseEnterThePromptForVideoGeneration') + '，'+ t('describeTheDigitalHumanImageBackgroundStyleActionRequirements');
            } else if (selectedTaskId.value === 'animate') {
                return t('optional') + ' '+ t('pleaseEnterThePromptForVideoGeneration') + '，'+ t('describeTheContentActionRequirementsBasedOnTheImage');
            }
            return t('pleaseEnterThePromptForVideoGeneration') + '...';
        };

        const getStatusTextClass = (status) => {
            if (status === 'SUCCEED') return 'text-emerald-400';
            if (status === 'CREATED') return 'text-blue-400';
            if (status === 'PENDING') return 'text-yellow-400';
            if (status === 'RUNNING') return 'text-amber-400';
            if (status === 'FAILED') return 'text-red-400';
            if (status === 'CANCEL') return 'text-gray-400';
            return 'text-gray-400';
        };

        const getImagePreview = (base64Data) => {
            if (!base64Data) return '';
            return `data:image/jpeg;base64,${base64Data}`;
        };

        const getTaskInputUrl = async (taskId, key) => {
            // 优先从缓存获取
            const cachedUrl = getTaskFileUrlSync(taskId, key);
            if (cachedUrl) {
                console.log('getTaskInputUrl: 从缓存获取', { taskId, key, url: cachedUrl });
                return cachedUrl;
            }
            return await getTaskFileUrlFromApi(taskId, key);
        };

        const getTaskInputImage = async (task) => {

            if (!task || !task.inputs) {
                console.log('getTaskInputImage: 任务或输入为空', { task: task?.task_id, inputs: task?.inputs });
                return null;
            }

            const imageInputs = Object.keys(task.inputs).filter(key =>
                key.includes('image') ||
                task.inputs[key].toString().toLowerCase().match(/\.(jpg|jpeg|png|gif|bmp|webp)$/)
            );

            if (imageInputs.length > 0) {
                const firstImageKey = imageInputs[0];
                // 优先从缓存获取
                const cachedUrl = getTaskFileUrlSync(task.task_id, firstImageKey);
                if (cachedUrl) {
                    console.log('getTaskInputImage: 从缓存获取', { taskId: task.task_id, key: firstImageKey, url: cachedUrl });
                    return cachedUrl;
                }
                // 缓存没有则生成URL
                const url = await getTaskInputUrl(task.task_id, firstImageKey);
                console.log('getTaskInputImage: 生成URL', { taskId: task.task_id, key: firstImageKey, url });
                return url;
            }

            console.log('getTaskInputImage: 没有找到图片输入');
            return null;
        };

        const getTaskInputAudio = async (task) => {
            if (!task || !task.inputs) return null;

            // Directly use 'input_audio' key
            const audioKey = 'input_audio';
            if (!task.inputs[audioKey]) return null;

            // Always bypass cache and check API directly to detect directory type
            // This ensures we get the correct URL even if cache has invalid data
            let url = await getTaskFileUrlFromApi(task.task_id, audioKey);

            // If it's a directory (multi-person mode) or URL is null, try to get original_audio file
            if (!url) {
                console.log(`Audio input ${audioKey} is a directory (multi-person mode), trying to get original_audio file`);
                // Try to get original_audio file from directory
                // Try common extensions
                const extensions = ['wav', 'mp3', 'mp4', 'aac', 'ogg', 'm4a'];
                for (const ext of extensions) {
                    const originalAudioFilename = `original_audio.${ext}`;
                    url = await getTaskFileUrlFromApi(task.task_id, audioKey, originalAudioFilename);
                    if (url) {
                        console.log(`Found original audio file: ${originalAudioFilename}`);
                        break;
                    }
                }
            }

            return url;
        };

        const handleThumbnailError = (event) => {
            // 当输入图片加载失败时，显示默认图标
            const img = event.target;
            const parent = img.parentElement;
            parent.innerHTML = '<div class="w-full h-44 bg-laser-purple/20 flex items-center justify-center"><i class="fas fa-video text-gradient-icon text-xl"></i></div>';
        };

        const handleImageError = (event) => {
            // 当图片加载失败时，隐藏图片，显示文件名
            const img = event.target;
            img.style.display = 'none';
            // 文件名已经显示，不需要额外处理
        };

        const handleImageLoad = (event) => {
            // 当图片加载成功时，显示图片和下载按钮，隐藏文件名
            const img = event.target;
            img.style.display = 'block';
            // 显示下载按钮
            const downloadBtn = img.parentElement.querySelector('button');
            if (downloadBtn) {
                downloadBtn.style.display = 'block';
            }
            // 隐藏文件名span
            const span = img.parentElement.parentElement.querySelector('span');
            if (span) {
                span.style.display = 'none';
            }
        };

        const handleAudioError = (event) => {
            // 当音频加载失败时，隐藏音频控件和下载按钮，显示文件名
            const audio = event.target;
            audio.style.display = 'none';
            // 隐藏下载按钮
            const downloadBtn = audio.parentElement.querySelector('button');
            if (downloadBtn) {
                downloadBtn.style.display = 'none';
            }
            // 文件名已经显示，不需要额外处理
        };

        const handleAudioLoad = (event) => {
            // 当音频加载成功时，显示音频控件和下载按钮，隐藏文件名
            const audio = event.target;
            audio.style.display = 'block';
            // 显示下载按钮
            const downloadBtn = audio.parentElement.querySelector('button');
            if (downloadBtn) {
                downloadBtn.style.display = 'block';
            }
            // 隐藏文件名span
            const span = audio.parentElement.parentElement.querySelector('span');
            if (span) {
                span.style.display = 'none';
            }
        };

        // 监听currentPage变化，同步更新pageInput
        watch(currentTaskPage, (newPage) => {
            taskPageInput.value = newPage;
        });

        // 监听pagination变化，确保分页组件更新
        watch(pagination, (newPagination) => {
            console.log('pagination变化:', newPagination);
            if (newPagination && newPagination.total_pages) {
                // 确保当前页不超过总页数
                if (currentTaskPage.value > newPagination.total_pages) {
                    currentTaskPage.value = newPagination.total_pages;
                }
            }
        }, { deep: true });

        // 监听templateCurrentPage变化，同步更新templatePageInput
        watch(templateCurrentPage, (newPage) => {
            templatePageInput.value = newPage;
        });

        // 监听templatePagination变化，确保分页组件更新
        watch(templatePagination, (newPagination) => {
            console.log('templatePagination变化:', newPagination);
            if (newPagination && newPagination.total_pages) {
                // 确保当前页不超过总页数
                if (templateCurrentPage.value > newPagination.total_pages) {
                    templateCurrentPage.value = newPagination.total_pages;
                }
            }
        }, { deep: true });

        // 监听inspirationCurrentPage变化，同步更新inspirationPageInput
        watch(inspirationCurrentPage, (newPage) => {
            inspirationPageInput.value = newPage;
        });

        // 监听inspirationPagination变化，确保分页组件更新
        watch(inspirationPagination, (newPagination) => {
            console.log('inspirationPagination变化:', newPagination);
            if (newPagination && newPagination.total_pages) {
                // 确保当前页不超过总页数
                if (inspirationCurrentPage.value > newPagination.total_pages) {
                    inspirationCurrentPage.value = newPagination.total_pages;
                }
            }
        }, { deep: true });

        // 统一的初始化函数
        const init = async () => {
            try {
                // 0. 初始化主题
                initTheme();

                // 1. 加载模型和任务数据
                await loadModels();

                // 2. 从路由恢复或设置默认值
                const routeQuery = router.currentRoute.value?.query || {};
                const routeTaskType = routeQuery.taskType;
                const routeModel = routeQuery.model;
                const routeExpanded = routeQuery.expanded;

                if (routeTaskType && availableTaskTypes.value.includes(routeTaskType)) {
                    // 路由中有 taskType，恢复它
                    selectTask(routeTaskType);

                    if (routeModel && availableModelClasses.value.includes(routeModel)) {
                        // 路由中有 model，恢复它（会自动设置 stage）
                        selectModel(routeModel);
                    } else {
                        // 路由中没有 model 或 model 无效，选择第一个模型
                        const firstModel = availableModelClasses.value[0];
                        if (firstModel) {
                            selectModel(firstModel);
                        }
                    }
                } else {
                    // 路由中没有 taskType，设置默认值：s2v
                    const defaultTaskType = availableTaskTypes.value.includes('s2v') ? 's2v' : availableTaskTypes.value[0];
                    if (defaultTaskType) {
                        selectTask(defaultTaskType);

                        // 选择该任务下的第一个模型
                        const firstModel = availableModelClasses.value[0];
                        if (firstModel) {
                            selectModel(firstModel);
                        }
                    }
                }

                // 3. 恢复 expanded 状态（如果路由中有）
                if (routeExpanded === 'true') {
                    expandCreationArea();
                }

                // 4. 加载历史记录和素材库（异步，不阻塞首屏）
                refreshTasks(true);
                loadInspirationData(true);

                // 5. 加载历史记录和素材库文件（异步，不阻塞首屏）
                getPromptHistory();
                loadTaskFilesFromCache();
                loadTemplateFilesFromCache();

                // 异步加载模板数据，不阻塞首屏渲染
                setTimeout(() => {
                    loadImageAudioTemplates(true);
                }, 100);


                console.log('初始化完成:', {
                    currentUser: currentUser.value,
                    availableModels: models.value,
                    tasks: tasks.value,
                    inspirationItems: inspirationItems.value,
                    selectedTaskId: selectedTaskId.value,
                    selectedModel: selectedModel.value,
                    currentForm: {
                        model_cls: getCurrentForm().model_cls,
                        stage: getCurrentForm().stage
                    }
                });

            } catch (error) {
                console.error('初始化失败:', error);
                showAlert(t('initFailedPleaseRefresh'), 'danger');
            }
        };

        // 重置表单函数（保留模型选择，清空图片、音频和提示词）
        const resetForm = async (taskType) => {
            const currentForm = getCurrentForm();
            const currentModel = currentForm.model_cls;
            const currentStage = currentForm.stage;

            // 重置表单但保留模型和阶段
            switch (taskType) {
                case 't2v':
                    t2vForm.value = {
                        task: 't2v',
                        model_cls: currentModel,
                        stage: currentStage,
                        prompt: '',
                        seed: Math.floor(Math.random() * 1000000)
                    };
                    break;
                case 'i2v':
                    i2vForm.value = {
                        task: 'i2v',
                        model_cls: currentModel,
                        stage: currentStage,
                        imageFile: null,
                        prompt: '',
                        seed: Math.floor(Math.random() * 1000000)
                    };
                    // 直接清空i2v图片预览
                    i2vImagePreview.value = null;
                    // 清理图片文件输入框
                    const imageInput = document.querySelector('input[type="file"][accept="image/*"]');
                    if (imageInput) {
                        imageInput.value = '';
                    }
                    break;
                case 's2v':
                    s2vForm.value = {
                        task: 's2v',
                        model_cls: currentModel,
                        stage: currentStage,
                        imageFile: null,
                        audioFile: null,
                        prompt: 'Make the character speak in a natural way according to the audio.',
                        seed: Math.floor(Math.random() * 1000000)
                    };
                    break;
                case 'animate':
                    animateForm.value = {
                        task: 'animate',
                        model_cls: currentModel,
                        stage: currentStage,
                        imageFile: null,
                        videoFile: null,
                        prompt: '视频中的人在做动作',
                        seed: Math.floor(Math.random() * 1000000),
                        detectedFaces: []
                    };
                    // 直接清空animate图片和视频预览
                    animateImagePreview.value = null;
                    animateVideoPreview.value = null;
                    // 清理图片和视频文件输入框
                    const animateImageInput = document.querySelector('input[type="file"][accept="image/*"]');
                    if (animateImageInput) {
                        animateImageInput.value = '';
                    }
                    const animateVideoInput = document.querySelector('input[type="file"][data-role="video-input"]');
                    if (animateVideoInput) {
                        animateVideoInput.value = '';
                    }
                    break;
            }

            // 强制触发Vue响应式更新
            setCurrentImagePreview(null);
            setCurrentAudioPreview(null);
            await nextTick();
        };

        // 开始轮询任务状态
        const startPollingTask = (taskId) => {
            if (!pollingTasks.value.has(taskId)) {
                pollingTasks.value.add(taskId);
                console.log(`开始轮询任务状态: ${taskId}`);

                // 如果还没有轮询定时器，启动一个
                if (!pollingInterval.value) {
                    pollingInterval.value = setInterval(async () => {
                        await pollTaskStatuses();
                    }, 1000); // 每1秒轮询一次
                }
            }
        };

        // 停止轮询任务状态
        const stopPollingTask = (taskId) => {
            pollingTasks.value.delete(taskId);
            console.log(`停止轮询任务状态: ${taskId}`);

            // 如果没有任务需要轮询了，清除定时器
            if (pollingTasks.value.size === 0 && pollingInterval.value) {
                clearInterval(pollingInterval.value);
                pollingInterval.value = null;
                console.log('停止所有任务状态轮询');
            }
        };

        const refreshTaskFiles = (task) => {
            for (const [key, inputPath] of Object.entries(task.inputs)) {
                getTaskFileUrlFromApi(task.task_id, key).then(url => {
                    console.log('refreshTaskFiles: input', task.task_id, key, url);
                });
            }
            for (const [key, outputPath] of Object.entries(task.outputs)) {
                getTaskFileUrlFromApi(task.task_id, key).then(url => {
                    console.log('refreshTaskFiles: output', task.task_id, key, url);
                });
            }
        };

        // 轮询任务状态
        const pollTaskStatuses = async () => {
            if (pollingTasks.value.size === 0) return;

            try {
                const taskIds = Array.from(pollingTasks.value);
                const response = await apiRequest(`/api/v1/task/query?task_ids=${taskIds.join(',')}`);

                if (response && response.ok) {
                    const tasksData = await response.json();
                    const updatedTasks = tasksData.tasks || [];

                    // 更新任务列表中的任务状态
                    let hasUpdates = false;
                    updatedTasks.forEach(updatedTask => {
                        const existingTaskIndex = tasks.value.findIndex(t => t.task_id === updatedTask.task_id);
                        if (existingTaskIndex !== -1) {
                            const oldTask = tasks.value[existingTaskIndex];
                            tasks.value[existingTaskIndex] = updatedTask;
                            console.log('updatedTask', updatedTask);
                            console.log('oldTask', oldTask);

                            // 如果状态发生变化，记录日志
                            if (oldTask !== updatedTask) {
                                hasUpdates = true; // 这里基本都会变，因为任务有进度条

                                // 如果当前在查看这个任务的详情，更新selectedTask
                                if (modalTask.value && modalTask.value.task_id === updatedTask.task_id) {
                                    modalTask.value = updatedTask;
                                    if (updatedTask.status === 'SUCCEED') {
                                        console.log('refresh viewing task: output files');
                                        loadTaskFiles(updatedTask);
                                    }
                                }

                                // 如果当前TaskCarousel显示的是这个任务，更新currentTask
                                if (currentTask.value && currentTask.value.task_id === updatedTask.task_id) {
                                    currentTask.value = updatedTask;
                                    console.log('TaskCarousel: 更新currentTask', updatedTask);
                                }

                                // 如果当前在projects页面且变化的是状态，更新tasks
                                if (router.path === '/projects' && oldTask.status !== updatedTask.status) {
                                    refreshTasks(true);
                                }

                                // 如果任务完成或失败，停止轮询并显示提示
                                if (['SUCCEED', 'FAILED', 'CANCEL'].includes(updatedTask.status)) {
                                    stopPollingTask(updatedTask.task_id);
                                    refreshTaskFiles(updatedTask);
                                    refreshTasks(true);

                                    // 显示任务完成提示
                                    if (updatedTask.status === 'SUCCEED') {
                                        showAlert(t('taskCompletedSuccessfully'), 'success', {
                                            label: t('view'),
                                            onClick: () => {
                                                openTaskDetailModal(updatedTask);
                                            }
                                        });
                                    } else if (updatedTask.status === 'FAILED') {
                                        showAlert(t('videoGeneratingFailed'), 'danger', {
                                            label: t('view'),
                                            onClick: () => {
                                                openTaskDetailModal(updatedTask);
                                            }
                                        });
                                    } else if (updatedTask.status === 'CANCEL') {
                                        showAlert(t('taskCancelled'), 'warning');
                                    }
                                }
                            }
                        }
                    });

                    // 如果有更新，触发界面刷新
                    if (hasUpdates) {
                        await nextTick();
                    }
                }
            } catch (error) {
                console.error('轮询任务状态失败:', error);
            }
        };

        // 任务状态管理
        const getTaskStatusDisplay = (status) => {
            const statusMap = {
                'CREATED': t('created'),
                'PENDING': t('pending'),
                'RUNNING': t('running'),
                'SUCCEED': t('succeed'),
                'FAILED': t('failed'),
                'CANCEL': t('cancelled')
            };
            return statusMap[status] || status;
        };

        const getTaskStatusColor = (status) => {
            const colorMap = {
                'CREATED': 'text-blue-400',
                'PENDING': 'text-yellow-400',
                'RUNNING': 'text-amber-400',
                'SUCCEED': 'text-emerald-400',
                'FAILED': 'text-red-400',
                'CANCEL': 'text-gray-400'
            };
            return colorMap[status] || 'text-gray-400';
        };

        const getTaskStatusIcon = (status) => {
            const iconMap = {
                'CREATED': 'fas fa-clock',
                'PENDING': 'fas fa-hourglass-half',
                'RUNNING': 'fas fa-spinner fa-spin',
                'SUCCEED': 'fas fa-check-circle',
                'FAILED': 'fas fa-exclamation-triangle',
                'CANCEL': 'fas fa-ban'
            };
            return iconMap[status] || 'fas fa-question-circle';
        };

        // 任务时间格式化
        const getTaskDuration = (startTime, endTime) => {
            if (!startTime || !endTime) return '未知';
            const start = new Date(startTime * 1000);
            const end = new Date(endTime * 1000);
            const diff = end - start;
            const minutes = Math.floor(diff / 60000);
            const seconds = Math.floor((diff % 60000) / 1000);
            return `${minutes}分${seconds}秒`;
        };

        // 相对时间格式化
        const getRelativeTime = (timestamp) => {
            if (!timestamp) return '未知';
            const now = new Date();
            const time = new Date(timestamp * 1000);
            const diff = now - time;

            const minutes = Math.floor(diff / 60000);
            const hours = Math.floor(diff / 3600000);
            const days = Math.floor(diff / 86400000);
            const months = Math.floor(diff / 2592000000); // 30天
            const years = Math.floor(diff / 31536000000);

            if (years > 0) {
                return years === 1 ? t('oneYearAgo') : `${years}t('yearsAgo')`;
            } else if (months > 0) {
                return months === 1 ? t('oneMonthAgo') : `${months}${t('monthsAgo')}`;
            } else if (days > 0) {
                return days === 1 ? t('oneDayAgo') : `${days}${t('daysAgo')}`;
            } else if (hours > 0) {
                return hours === 1 ? t('oneHourAgo') : `${hours}${t('hoursAgo')}`;
            } else if (minutes > 0) {
                return minutes === 1 ? t('oneMinuteAgo') : `${minutes}${t('minutesAgo')}`;
            } else {
                return t('justNow');
            }
        };

        // 任务历史记录管理
        const getTaskHistory = () => {
            return tasks.value.filter(task =>
                ['SUCCEED', 'FAILED', 'CANCEL'].includes(task.status)
            );
        };

        // 子任务进度相关函数
        const getOverallProgress = (subtasks) => {
            if (!subtasks || subtasks.length === 0) return 0;

            let completedCount = 0;
            subtasks.forEach(subtask => {
                if (subtask.status === 'SUCCEED') {
                    completedCount++;
                }
            });

            return Math.round((completedCount / subtasks.length) * 100);
        };

        // 获取进度条标题
        const getProgressTitle = (subtasks) => {
            if (!subtasks || subtasks.length === 0) return t('overallProgress');

            const pendingSubtasks = subtasks.filter(subtask => subtask.status === 'PENDING');
            const runningSubtasks = subtasks.filter(subtask => subtask.status === 'RUNNING');

            if (pendingSubtasks.length > 0) {
                return t('queueStatus');
            } else if (runningSubtasks.length > 0) {
                return t('running');
            } else {
                return t('overallProgress');
            }
        };

        // 获取进度信息
        const getProgressInfo = (subtasks) => {
            if (!subtasks || subtasks.length === 0) return '0%';

            const pendingSubtasks = subtasks.filter(subtask => subtask.status === 'PENDING');
            const runningSubtasks = subtasks.filter(subtask => subtask.status === 'RUNNING');

            if (pendingSubtasks.length > 0) {
                // 显示排队信息
                const firstPending = pendingSubtasks[0];
                const queuePosition = firstPending.estimated_pending_order;
                const estimatedTime = firstPending.estimated_pending_secs;

                let info = t('queueing');
                if (queuePosition !== null && queuePosition !== undefined) {
                    info += ` (${t('position')}: ${queuePosition})`;
                }
                if (estimatedTime !== null && estimatedTime !== undefined) {
                    info += ` - ${formatDuration(estimatedTime)}`;
                }
                return info;
            } else if (runningSubtasks.length > 0) {
                // 显示运行信息
                const firstRunning = runningSubtasks[0];
                const workerName = firstRunning.worker_name || t('unknown');
                const estimatedTime = firstRunning.estimated_running_secs;

                let info = `${t('subtask')} ${workerName}`;
                if (estimatedTime !== null && estimatedTime !== undefined) {
                    const elapses = firstRunning.elapses || {};
                    const runningTime = elapses['RUNNING-'] || 0;
                    const remaining = Math.max(0, estimatedTime - runningTime);
                    info += ` - ${t('remaining')} ${formatDuration(remaining)}`;
                }
                return info;
            } else {
                // 显示总体进度
                return getOverallProgress(subtasks) + '%';
            }
        };

        const getSubtaskProgress = (subtask) => {
            if (subtask.status === 'SUCCEED') return 100;
            if (subtask.status === 'FAILED' || subtask.status === 'CANCEL') return 0;

            // 对于PENDING和RUNNING状态，基于时间估算进度
            if (subtask.status === 'PENDING') {
                // 排队中的任务，进度为0
                return 0;
            }

            if (subtask.status === 'RUNNING') {
                // 运行中的任务，基于已运行时间估算进度
                const elapses = subtask.elapses || {};
                const runningTime = elapses['RUNNING-'] || 0;
                const estimatedTotal = subtask.estimated_running_secs || 0;

                if (estimatedTotal > 0) {
                    const progress = Math.min((runningTime / estimatedTotal) * 100, 95); // 最多95%，避免显示100%但未完成
                    return Math.round(progress);
                }

                // 如果没有时间估算，基于状态显示一个基础进度
                return 50; // 运行中但无法估算进度时显示50%
            }

            return 0;
        };



        const getSubtaskStatusText = (status) => {
            const statusMap = {
                'PENDING': t('pending'),
                'RUNNING': t('running'),
                'SUCCEED': t('completed'),
                'FAILED': t('failed'),
                'CANCEL': t('cancelled')
            };
            return statusMap[status] || status;
        };


        const formatEstimatedTime = computed(() => {
            return (formattedEstimatedTime) => {
            if (subtask.status === 'PENDING') {
                const pendingSecs = subtask.estimated_pending_secs;
                const queuePosition = subtask.estimated_pending_order;

                if (pendingSecs !== null && pendingSecs !== undefined) {
                    let info = formatDuration(pendingSecs);
                    if (queuePosition !== null && queuePosition !== undefined) {
                        info += ` (${t('position')}: ${queuePosition})`;
                    }
                    formattedEstimatedTime.value = info;
                }
                formattedEstimatedTime.value=t('calculating');
            }

            if (subtask.status === 'RUNNING') {
                // 使用extra_info.elapses而不是subtask.elapses
                const elapses = subtask.extra_info?.elapses || {};
                const runningTime = elapses['RUNNING-'] || 0;
                const estimatedTotal = subtask.estimated_running_secs || 0;

                if (estimatedTotal > 0) {
                    const remaining = Math.max(0, estimatedTotal - runningTime);
                    estimatedTime.value = remaining;
                    formattedEstimatedTime.value = `${t('remaining')} ${formatDuration(remaining)}`;
                }

                // 如果没有estimated_running_secs，尝试使用elapses计算
                if (Object.keys(elapses).length > 0) {
                    const totalElapsed = Object.values(elapses).reduce((sum, time) => sum + (time || 0), 0);
                    if (totalElapsed > 0) {
                        formattedEstimatedTime.value = `${t('running')} ${formatDuration(totalElapsed)}`;
                    }
                }

                return t('calculating');
            }

            return t('completed');
        };
});

        const formatDuration = (seconds) => {
            if (seconds < 60) {
                return `${Math.round(seconds)}${t('seconds')}`;
            } else if (seconds < 3600) {
                const minutes = Math.floor(seconds / 60);
                const remainingSeconds = Math.round(seconds % 60);
                return `${minutes}${t('minutes')}${remainingSeconds}${t('seconds')}`;
            } else {
                const hours = Math.floor(seconds / 3600);
                const minutes = Math.floor((seconds % 3600) / 60);
                const remainingSeconds = Math.round(seconds % 60);
                return `${hours}${t('hours')}${minutes}${t('minutes')}${remainingSeconds}${t('seconds')}`;
            }
        };

        const getActiveTasks = () => {
            return tasks.value.filter(task =>
                ['CREATED', 'PENDING', 'RUNNING'].includes(task.status)
            );
        };

        // 任务搜索和过滤增强
        const searchTasks = (query) => {
            if (!query) return tasks.value;
            return tasks.value.filter(task => {
                const searchText = [
                    task.task_id,
                    task.task_type,
                    task.model_cls,
                    task.params?.prompt || '',
                    getTaskStatusDisplay(task.status)
                ].join(' ').toLowerCase();
                return searchText.includes(query.toLowerCase());
            });
        };

        const filterTasksByStatus = (status) => {
            if (status === 'ALL') return tasks.value;
            return tasks.value.filter(task => task.status === status);
        };

        const filterTasksByType = (type) => {
            if (!type) return tasks.value;
            return tasks.value.filter(task => task.task_type === type);
        };

        // 提示消息样式管理
        const getAlertClass = (type) => {
            const classMap = {
                'success': 'animate-slide-down',
                'warning': 'animate-slide-down',
                'danger': 'animate-slide-down',
                'info': 'animate-slide-down'
            };
            return classMap[type] || 'animate-slide-down';
        };

        const getAlertBorderClass = (type) => {
            const borderMap = {
                'success': 'border-green-500',
                'warning': 'border-yellow-500',
                'danger': 'border-red-500',
                'info': 'border-blue-500'
            };
            return borderMap[type] || 'border-gray-500';
        };

        const getAlertTextClass = (type) => {
            // 统一使用白色文字
            return 'text-white';
        };

        const getAlertIcon = (type) => {
            const iconMap = {
                'success': 'fas fa-check text-white',
                'warning': 'fas fa-exclamation text-white',
                'danger': 'fas fa-times text-white',
                'info': 'fas fa-info text-white'
            };
            return iconMap[type] || 'fas fa-info text-white';
        };

        const getAlertIconBgClass = (type) => {
            const bgMap = {
                'success': 'bg-green-500/30',
                'warning': 'bg-yellow-500/30',
                'danger': 'bg-red-500/30',
                'info': 'bg-laser-purple/30'
            };
            return bgMap[type] || 'bg-laser-purple/30';
        };

        // 监听器 - 监听任务类型变化
        watch(() => selectedTaskId.value, () => {
            const currentForm = getCurrentForm();

            // 只有当当前表单没有选择模型时，才自动选择第一个可用的模型
            if (!currentForm.model_cls) {
                let availableModels;

                availableModels = models.value.filter(m => m.task === selectedTaskId.value);

                if (availableModels.length > 0) {
                    const firstModel = availableModels[0];
                    currentForm.model_cls = firstModel.model_cls;
                    currentForm.stage = firstModel.stage;
                }
            }

            // 注意：这里不需要重置预览，因为我们要保持每个任务的独立性
            // 预览会在 selectTask 函数中根据文件状态恢复
        });

        watch(() => getCurrentForm().model_cls, () => {
            const currentForm = getCurrentForm();

            // 只有当当前表单没有选择阶段时，才自动选择第一个可用的阶段
            if (!currentForm.stage) {
                let availableStages;

                availableStages = models.value
                        .filter(m => m.task === selectedTaskId.value && m.model_cls === currentForm.model_cls)
                        .map(m => m.stage);

                if (availableStages.length > 0) {
                    currentForm.stage = availableStages[0];
                }
            }
        });

        // 提示词模板管理
        const promptTemplates = {
            's2v': [
                {
                    id: 's2v_1',
                    title: '情绪表达',
                    prompt: '根据音频，人物进行情绪化表达，表情丰富，能体现音频中的情绪，手势根据情绪适当调整。'
                },
                {
                    id: 's2v_2',
                    title: '故事讲述',
                    prompt: '根据音频，人物进行故事讲述，表情丰富，能体现音频中的情绪，手势根据故事情节适当调整。'
                },
                {
                    id: 's2v_3',
                    title: '知识讲解',
                    prompt: '根据音频，人物进行知识讲解，表情严肃，整体风格专业得体，手势根据知识内容适当调整。'
                },
                {
                    id: 's2v_4',
                    title: '浮夸表演',
                    prompt: '根据音频，人物进行浮夸表演，表情夸张，动作浮夸，整体风格夸张搞笑。'
                },
                {
                    id: 's2v_5',
                    title: '商务演讲',
                    prompt: '根据音频，人物进行商务演讲，表情严肃，手势得体，整体风格专业商务。'
                },
                {
                    id: 's2v_6',
                    title: '产品介绍',
                    prompt: '数字人介绍产品特点，语气亲切热情，表情丰富，动作自然，能体现产品特点。'
                }
            ],
            't2v': [
                {
                    id: 't2v_1',
                    title: '自然风景',
                    prompt: '一个宁静的山谷，阳光透过云层洒在绿色的草地上，远处有雪山，近处有清澈的溪流，画面温暖自然，充满生机。'
                },
                {
                    id: 't2v_2',
                    title: '城市夜景',
                    prompt: '繁华的城市夜景，霓虹灯闪烁，高楼大厦林立，车流如织，天空中有星星点缀，营造出都市的繁华氛围。'
                },
                {
                    id: 't2v_3',
                    title: '科技未来',
                    prompt: '未来科技城市，飞行汽车穿梭，全息投影随处可见，建筑具有流线型设计，充满科技感和未来感。'
                }
            ],
            'i2v': [
                {
                    id: 'i2v_1',
                    title: '人物动作',
                    prompt: '基于参考图片，让角色做出自然的行走动作，保持原有的服装和风格，背景可以适当变化。'
                },
                {
                    id: 'i2v_2',
                    title: '场景转换',
                    prompt: '保持参考图片中的人物形象，将背景转换为不同的季节或环境，如从室内到户外，从白天到夜晚。'
                }
            ]
        };

        const getPromptTemplates = (taskType) => {
            return promptTemplates[taskType] || [];
        };

        const selectPromptTemplate = (template) => {
            getCurrentForm().prompt = template.prompt;
            showPromptModal.value = false;
            showAlert(`${t('templateApplied')} ${template.title}`, 'success');
        };

        // 提示词历史记录管理 - 现在直接从taskHistory中获取
        const promptHistory = ref([]);

        const getPromptHistory = async () => {
            try {
                // 从taskHistory中获取prompt历史，去重并按时间排序
                const taskHistory = await getLocalTaskHistory();
                const uniquePrompts = [];
                const seenPrompts = new Set();

                // 遍历taskHistory，提取唯一的prompt
                for (const task of taskHistory) {
                    if (task.prompt && task.prompt.trim() && !seenPrompts.has(task.prompt.trim())) {
                        uniquePrompts.push(task.prompt.trim());
                        seenPrompts.add(task.prompt.trim());
                    }
                }

                const result = uniquePrompts.slice(0, 10); // 只显示最近10条
                promptHistory.value = result; // 更新响应式数据
                return result;
            } catch (error) {
                console.error(t('getPromptHistoryFailed'), error);
                promptHistory.value = []; // 更新响应式数据
                return [];
            }
        };

        // addPromptToHistory函数已删除，现在prompt历史直接从taskHistory中获取

        // 保存完整的任务历史（只保存元数据，不保存文件内容）
        const addTaskToHistory = (taskType, formData) => {
            console.log('开始保存任务历史:', { taskType, formData });

            const historyItem = {
                id: Date.now(),
                timestamp: new Date().toISOString(),
                taskType: taskType,
                prompt: formData.prompt || '',
                // 只保存文件元数据，不保存文件内容
                imageFile: formData.imageFile ? {
                        name: formData.imageFile.name,
                        type: formData.imageFile.type,
                    size: formData.imageFile.size
                    // 不再保存 data 字段，避免占用大量存储空间
                } : null,
                audioFile: formData.audioFile ? {
                        name: formData.audioFile.name,
                        type: formData.audioFile.type,
                    size: formData.audioFile.size
                    // 不再保存 data 字段，避免占用大量存储空间
                } : null
                    };

            console.log('保存任务历史（仅元数据）:', historyItem);
                saveTaskHistoryItem(historyItem);
        };

        // 保存任务历史项到localStorage
        const saveTaskHistoryItem = (historyItem) => {
            try {
                const existingHistory = JSON.parse(localStorage.getItem('taskHistory') || '[]');

                // 避免重复添加（基于提示词、任务类型、图片和音频）
                const isDuplicate = existingHistory.some(item => {
                    const samePrompt = item.prompt === historyItem.prompt;
                    const sameTaskType = item.taskType === historyItem.taskType;
                    const sameImage = (item.imageFile?.name || '') === (historyItem.imageFile?.name || '');
                    const sameAudio = (item.audioFile?.name || '') === (historyItem.audioFile?.name || '');

                    return samePrompt && sameTaskType && sameImage && sameAudio;
                });

                if (!isDuplicate) {
                    // 按时间戳排序，确保最新的记录在最后
                    existingHistory.push(historyItem);
                    existingHistory.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

                    // 限制历史记录数量为10条（不再保存文件内容，所以可以适当减少）
                    if (existingHistory.length > 10) {
                        existingHistory.splice(0, existingHistory.length - 10);
                    }

                    // 保存到localStorage
                    try {
                        localStorage.setItem('taskHistory', JSON.stringify(existingHistory));
                        console.log('任务历史已保存（仅元数据）:', historyItem);
                    } catch (storageError) {
                        if (storageError.name === 'QuotaExceededError') {
                            console.warn('localStorage空间不足，尝试清理旧数据...');

                            // 清理策略1：只保留最新的5条记录
                            const cleanedHistory = existingHistory.slice(-5);

                            try {
                                localStorage.setItem('taskHistory', JSON.stringify(cleanedHistory));
                                console.log('任务历史已保存（清理后）:', historyItem);
                            } catch (secondError) {
                                console.error('清理后仍无法保存，尝试清理所有缓存...');

                                // 清理策略2：清理所有任务历史，只保存当前这一条
                                try {
                                    localStorage.setItem('taskHistory', JSON.stringify([historyItem]));
                                    console.log('任务历史已保存（完全清理后）');
                                    showAlert(t('historyCleared'), 'info');
                                } catch (thirdError) {
                                    console.error('即使完全清理后仍无法保存:', thirdError);
                                    // 不再显示警告，因为历史记录不是必需的功能
                                    console.warn('历史记录功能暂时不可用，将从任务列表恢复数据');
                                }
                            }
                        } else {
                            throw storageError;
                        }
                    }
                } else {
                    console.log('任务历史重复，跳过保存:', historyItem);
                }
            } catch (error) {
                console.error('保存任务历史失败:', error);
                // 不再显示警告给用户，因为可以从任务列表恢复数据
                console.warn('历史记录保存失败，将依赖任务列表数据');
            }
        };

        // 获取本地存储的任务历史
        const getLocalTaskHistory = async () => {
            try {
                // 使用Promise模拟异步操作，避免阻塞UI
                return await new Promise((resolve) => {
                    setTimeout(() => {
                        try {
                            const history = JSON.parse(localStorage.getItem('taskHistory') || '[]');
                            // 按时间戳排序，最新的记录在前
                            const sortedHistory = history.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
                            resolve(sortedHistory);
                        } catch (error) {
                            console.error(t('parseTaskHistoryFailed'), error);
                            resolve([]);
                        }
                    }, 0);
                });
            } catch (error) {
                console.error(t('getTaskHistoryFailed'), error);
                return [];
            }
        };

        const selectPromptHistory = (prompt) => {
            getCurrentForm().prompt = prompt;
            showPromptModal.value = false;
            showAlert(t('promptHistoryApplied'), 'success');
        };

        const clearPromptHistory = () => {
            // 清空taskHistory中的prompt相关数据
            localStorage.removeItem('taskHistory');
            showAlert(t('promptHistoryCleared'), 'info');
        };

        // 图片历史记录管理 - 从任务列表获取
        const getImageHistory = async () => {
            try {
                // 确保任务列表已加载
                if (tasks.value.length === 0) {
                    await refreshTasks();
                }

                const uniqueImages = [];
                const seenImages = new Set();

                // 遍历任务列表，提取唯一的图片
                for (const task of tasks.value) {
                    if (task.inputs && task.inputs.input_image && !seenImages.has(task.inputs.input_image)) {
                        // 获取图片URL
                        const imageUrl = await getTaskFileUrl(task.task_id, 'input_image');
                        if (imageUrl) {
                            uniqueImages.push({
                                filename: task.inputs.input_image,
                                url: imageUrl,
                                thumbnail: imageUrl, // 使用URL作为缩略图
                                taskId: task.task_id,
                                timestamp: task.create_t,
                                taskType: task.task_type
                            });
                            seenImages.add(task.inputs.input_image);
                        }
                    }
                }

                // 按时间戳排序，最新的在前
                uniqueImages.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

                imageHistory.value = uniqueImages;
                console.log('从任务列表获取图片历史:', uniqueImages.length, '条');
                return uniqueImages;
            } catch (error) {
                console.error('获取图片历史失败:', error);
                imageHistory.value = [];
                return [];
            }
        };

        // 音频历史记录管理 - 从任务列表获取
        const getAudioHistory = async () => {
            try {
                // 确保任务列表已加载
                if (tasks.value.length === 0) {
                    await refreshTasks();
                }

                const uniqueAudios = [];
                const seenAudios = new Set();

                // 遍历任务列表，提取唯一的音频
                for (const task of tasks.value) {
                    if (task.inputs && task.inputs.input_audio && !seenAudios.has(task.inputs.input_audio)) {
                        // 获取音频URL
                        let audioUrl = await getTaskFileUrl(task.task_id, 'input_audio');

                        // 如果返回null，可能是目录类型（多人模式），尝试获取original_audio.wav
                        if (!audioUrl) {
                            audioUrl = await getTaskFileUrlFromApi(task.task_id, 'input_audio', 'original_audio.wav');
                        }

                        const imageUrl = task.inputs.input_image ? await getTaskFileUrl(task.task_id, 'input_image') : null;
                        if (audioUrl) {
                            uniqueAudios.push({
                                filename: task.inputs.input_audio,
                                url: audioUrl,
                                taskId: task.task_id,
                                timestamp: task.create_t,
                                taskType: task.task_type,
                                imageUrl
                            });
                            seenAudios.add(task.inputs.input_audio);
                        }
                    }
                }

                // 按时间戳排序，最新的在前
                uniqueAudios.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

                audioHistory.value = uniqueAudios;
                console.log('从任务列表获取音频历史:', uniqueAudios.length, '条');
                return uniqueAudios;
            } catch (error) {
                console.error('获取音频历史失败:', error);
                audioHistory.value = [];
                return [];
            }
        };

        // 选择图片历史记录 - 从URL获取
        const selectImageHistory = async (history) => {
            try {
                // 确保 URL 有效，如果无效则重新获取
                let imageUrl = history.url;
                if (!imageUrl || imageUrl.trim() === '') {
                    // 如果 URL 为空，尝试重新获取
                    if (history.taskId) {
                        imageUrl = await getTaskFileUrl(history.taskId, 'input_image');
                    }
                    if (!imageUrl || imageUrl.trim() === '') {
                        throw new Error('图片 URL 无效');
                    }
                }

                // 从URL获取图片文件
                const response = await fetch(imageUrl);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const blob = await response.blob();
                const file = new File([blob], history.filename, { type: blob.type });

                // 设置图片预览
                setCurrentImagePreview(imageUrl);
                updateUploadedContentStatus();

                // 更新表单
                const currentForm = getCurrentForm();
                currentForm.imageFile = file;

                // Reset detected faces
                if (selectedTaskId.value === 'i2v') {
                    i2vForm.value.detectedFaces = [];
                } else if (selectedTaskId.value === 's2v') {
                    s2vForm.value.detectedFaces = [];
                }

                showImageTemplates.value = false;
                showAlert(t('historyImageApplied'), 'success');

                // Auto detect faces after image is loaded
                // 不再自动检测人脸，等待用户手动打开多角色模式开关
                try {
                    // 如果 URL 是 http/https，直接使用；否则转换为 data URL
                    if (!imageUrl.startsWith('http://') && !imageUrl.startsWith('https://')) {
                        // 如果不是 http/https URL，转换为 data URL
                        const reader = new FileReader();
                        reader.onload = async (e) => {
                            // 不再自动检测人脸
                        };
                        reader.readAsDataURL(file);
                    }
                } catch (error) {
                    console.error('Face detection failed:', error);
                    // Don't show error alert, just log it
                }

            } catch (error) {
                console.error('应用历史图片失败:', error);
                showAlert(t('applyHistoryImageFailed') + ': ' + error.message, 'danger');
            }
        };

        // 选择音频历史记录 - 从URL获取
        const selectAudioHistory = async (history) => {
            try {
                // 从URL获取音频文件
                const response = await fetch(history.url);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const blob = await response.blob();
                const file = new File([blob], history.filename, { type: blob.type });

                // 设置音频预览
                setCurrentAudioPreview(history.url);
                updateUploadedContentStatus();

                // 更新表单
                const currentForm = getCurrentForm();
                currentForm.audioFile = file;

                showAudioTemplates.value = false;
                showAlert(t('historyAudioApplied'), 'success');
            } catch (error) {
                console.error('应用历史音频失败:', error);
                showAlert(t('applyHistoryAudioFailed'), 'danger');
            }
        };

        // 全局音频播放状态管理
        let currentPlayingAudio = null;
        let audioStopCallback = null;

        // 停止音频播放
        const stopAudioPlayback = () => {
            if (currentPlayingAudio) {
                currentPlayingAudio.pause();
                currentPlayingAudio.currentTime = 0;
                currentPlayingAudio = null;

                // 调用停止回调
                if (audioStopCallback) {
                    audioStopCallback();
                    audioStopCallback = null;
                }
            }
        };

        // 设置音频停止回调
        const setAudioStopCallback = (callback) => {
            audioStopCallback = callback;
        };

        // 预览音频历史记录 - 使用URL
        const previewAudioHistory = (history) => {
            console.log('预览音频历史:', history);
            const audioUrl = history.url;
            console.log('音频历史URL:', audioUrl);
            if (!audioUrl) {
                showAlert(t('audioHistoryUrlFailed'), 'danger');
                return;
            }

            // 停止当前播放的音频
            if (currentPlayingAudio) {
                currentPlayingAudio.pause();
                currentPlayingAudio.currentTime = 0;
                currentPlayingAudio = null;
            }

            const audio = new Audio(audioUrl);
            currentPlayingAudio = audio;

            // 监听音频播放结束事件
            audio.addEventListener('ended', () => {
                currentPlayingAudio = null;
                // 调用停止回调
                if (audioStopCallback) {
                    audioStopCallback();
                    audioStopCallback = null;
                }
            });

            audio.addEventListener('error', () => {
                console.error('音频播放失败:', audio.error);
                showAlert(t('audioPlaybackFailed'), 'danger');
                currentPlayingAudio = null;
                // 调用停止回调
                if (audioStopCallback) {
                    audioStopCallback();
                    audioStopCallback = null;
                }
            });

            audio.play().catch(error => {
                console.error('音频播放失败:', error);
                showAlert(t('audioPlaybackFailed'), 'danger');
                currentPlayingAudio = null;
            });
        };

        // 清空图片历史记录
        const clearImageHistory = () => {
            imageHistory.value = [];
            showAlert(t('imageHistoryCleared'), 'info');
        };

        // 清空音频历史记录
        const clearAudioHistory = () => {
            audioHistory.value = [];
            showAlert(t('audioHistoryCleared'), 'info');
        };

        // 清理localStorage存储空间
        const clearLocalStorage = () => {
            try {
                // 清理任务历史
                localStorage.removeItem('taskHistory');
                localStorage.removeItem('refreshToken');

                // 清理其他可能的缓存数据
                const keysToRemove = [];
                for (let i = 0; i < localStorage.length; i++) {
                    const key = localStorage.key(i);
                    if (key && (key.includes('template') || key.includes('task') || key.includes('history'))) {
                        keysToRemove.push(key);
                    }
                }

                keysToRemove.forEach(key => {
                    localStorage.removeItem(key);
                });

                // 重置相关状态
                imageHistory.value = [];
                audioHistory.value = [];
                promptHistory.value = [];

                showAlert(t('storageCleared'), 'success');
                console.log('localStorage已清理，释放了存储空间');
            } catch (error) {
                console.error('清理localStorage失败:', error);
                showAlert(t('clearStorageFailed'), 'danger');
            }
        };

        const getAuthHeaders = () => {
            const headers = {
                'Content-Type': 'application/json'
            };

            const token = localStorage.getItem('accessToken');
            if (token) {
                headers['Authorization'] = `Bearer ${token}`;
                console.log('使用Token进行认证:', token.substring(0, 20) + '...');
            } else {
                console.warn('没有找到accessToken');
            }
            return headers;
        };

        // 验证token是否有效
        const validateToken = async (token) => {
            try {
                const response = await fetch('/api/v1/model/list', {
                    method: 'GET',
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json'
                    }
                });
                await new Promise(resolve => setTimeout(resolve, 100));
                return response.ok;
            } catch (error) {
                console.error('Token validation failed:', error);
                return false;
            }
        };

        const refreshAccessToken = async () => {
            if (refreshPromise) {
                return refreshPromise;
            }
            const refreshToken = localStorage.getItem('refreshToken');
            if (!refreshToken) {
                return false;
            }

            refreshPromise = (async () => {
                try {
                    const response = await fetch('/auth/refresh', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ refresh_token: refreshToken })
                    });
                    await new Promise(resolve => setTimeout(resolve, 100));
                    if (!response.ok) {
                        throw new Error(`Refresh failed with status ${response.status}`);
                    }

                    const data = await response.json();
                    if (data.access_token) {
                        localStorage.setItem('accessToken', data.access_token);
                    }
                    if (data.refresh_token) {
                        localStorage.setItem('refreshToken', data.refresh_token);
                    }
                    if (data.user_info) {
                        currentUser.value = data.user_info;
                        localStorage.setItem('currentUser', JSON.stringify(data.user_info));
                    }
                    return true;
                } catch (error) {
                    console.error('Refresh token failed:', error);
                    logout(false);
                    showAlert(t('loginExpiredPleaseRelogin'), 'warning', {
                        label: t('login'),
                        onClick: login
                    });
                    return false;
                } finally {
                    refreshPromise = null;
                }
            })();

            return refreshPromise;
        };

        // 增强的API请求函数，自动处理认证错误
        const apiRequest = async (url, options = {}, allowRetry = true) => {
            const headers = getAuthHeaders();

            try {
                const response = await fetch(url, {
                    ...options,
                    headers: {
                        ...headers,
                        ...options.headers
                    }
                });
                await new Promise(resolve => setTimeout(resolve, 100));
                // 检查是否是认证错误
                if ((response.status === 401 || response.status === 403) && allowRetry) {
                    const refreshed = await refreshAccessToken();
                    if (refreshed) {
                        return await apiRequest(url, options, false);
                    }
                    return null;
                }

                return response;
            } catch (error) {
                console.error('API request failed:', error);
                showAlert(t('networkRequestFailed'), 'danger');
                return null;
            }
        };

        // 侧边栏拖拽调整功能
        const sidebar = ref(null);
        const sidebarWidth = ref(256); // 默认宽度 256px (w-64)
        let isResizing = false;
        let startX = 0;
        let startWidth = 0;

        // 更新悬浮按钮位置
        const updateFloatingButtonPosition = (width) => {
            const floatingBtn = document.querySelector('.floating-toggle-btn');
            if (floatingBtn) {
                if (sidebarCollapsed.value) {
                    // 收起状态时，按钮位于屏幕左侧
                    floatingBtn.style.left = '0px';
                    floatingBtn.style.right = 'auto';
                } else {
                    // 展开状态时，按钮位于历史任务栏右侧
                    floatingBtn.style.left = width + 'px';
                    floatingBtn.style.right = 'auto';
                }
            }
        };

        const startResize = (e) => {
            e.preventDefault();
            console.log('startResize called');

            isResizing = true;
            startX = e.clientX;
            startWidth = sidebar.value.offsetWidth;
            console.log('Resize started, width:', startWidth);

            document.body.classList.add('resizing');
            document.addEventListener('mousemove', handleResize);
            document.addEventListener('mouseup', stopResize);
        };

        const handleResize = (e) => {
            if (!isResizing) return;

            const deltaX = e.clientX - startX;
            const newWidth = startWidth + deltaX;
            const minWidth = 200;
            const maxWidth = 500;

            if (newWidth >= minWidth && newWidth <= maxWidth) {
                // 立即更新悬浮按钮位置，不等待其他更新
                const floatingBtn = document.querySelector('.floating-toggle-btn');
                if (floatingBtn && !sidebarCollapsed.value) {
                    floatingBtn.style.left = newWidth + 'px';
                }

                sidebarWidth.value = newWidth; // 更新响应式变量
                sidebar.value.style.setProperty('width', newWidth + 'px', 'important');

                // 同时调整主内容区域宽度
                const mainContent = document.querySelector('.main-container main');
                if (mainContent) {
                    mainContent.style.setProperty('width', `calc(100% - ${newWidth}px)`, 'important');
                } else {
                    const altMain = document.querySelector('main');
                    if (altMain) {
                        altMain.style.setProperty('width', `calc(100% - ${newWidth}px)`, 'important');
                    }
                }
            } else {
                console.log('Width out of range:', newWidth);
            }
        };

        const stopResize = () => {
            isResizing = false;
            document.body.classList.remove('resizing');
            document.removeEventListener('mousemove', handleResize);
            document.removeEventListener('mouseup', stopResize);

            // 保存当前宽度到localStorage
            if (sidebar.value) {
                localStorage.setItem('sidebarWidth', sidebar.value.offsetWidth);
            }
        };

        // 应用响应式侧边栏宽度
        const applyResponsiveWidth = () => {
            if (!sidebar.value) return;

            const windowWidth = window.innerWidth;
            let sidebarWidthPx;

            if (windowWidth <= 768) {
                sidebarWidthPx = 200;
            } else if (windowWidth <= 1200) {
                sidebarWidthPx = 250;
            } else {
                // 大屏幕时使用保存的宽度或默认宽度
                const savedWidth = localStorage.getItem('sidebarWidth');
                if (savedWidth) {
                    const width = parseInt(savedWidth);
                    if (width >= 200 && width <= 500) {
                        sidebarWidthPx = width;
                    } else {
                        sidebarWidthPx = 256; // 默认 w-64
                    }
                } else {
                    sidebarWidthPx = 256; // 默认 w-64
                }
            }

            sidebarWidth.value = sidebarWidthPx; // 更新响应式变量
            sidebar.value.style.width = sidebarWidthPx + 'px';

            // 更新悬浮按钮位置
            updateFloatingButtonPosition(sidebarWidthPx);

            const mainContent = document.querySelector('main');
            if (mainContent) {
                mainContent.style.width = `calc(100% - ${sidebarWidthPx}px)`;
            }
        };

        // 新增：视图切换方法
        const switchToCreateView = () => {
            // 生成页面的查询参数
            const generateQuery = {};

            // 保留任务类型选择
            if (selectedTaskId.value) {
                generateQuery.taskType = selectedTaskId.value;
            }

            // 保留模型选择
            if (selectedModel.value) {
                generateQuery.model = selectedModel.value;
            }

            // 保留创作区域展开状态
            if (isCreationAreaExpanded.value) {
                generateQuery.expanded = 'true';
            }

            router.push({ path: '/generate', query: generateQuery });

            // 如果之前有展开过创作区域，保持展开状态
            if (isCreationAreaExpanded.value) {
                // 延迟一点时间确保DOM更新完成
                setTimeout(() => {
                    const creationArea = document.querySelector('.creation-area');
                    if (creationArea) {
                        creationArea.classList.add('show');
                    }
                }, 50);
            }
        };

        const switchToProjectsView = (forceRefresh = false) => {
            // 项目页面的查询参数
            const projectsQuery = {};

            // 保留搜索查询
            if (taskSearchQuery.value) {
                projectsQuery.search = taskSearchQuery.value;
            }

            // 保留状态筛选
            if (statusFilter.value) {
                projectsQuery.status = statusFilter.value;
            }

            // 保留当前页码
            if (currentTaskPage.value > 1) {
                projectsQuery.page = currentTaskPage.value.toString();
            }

            router.push({ path: '/projects', query: projectsQuery });
            // 刷新任务列表
            refreshTasks(forceRefresh);
        };

        const switchToInspirationView = () => {
            // 灵感页面的查询参数
            const inspirationQuery = {};

            // 保留搜索查询
            if (inspirationSearchQuery.value) {
                inspirationQuery.search = inspirationSearchQuery.value;
            }

            // 保留分类筛选
            if (selectedInspirationCategory.value) {
                inspirationQuery.category = selectedInspirationCategory.value;
            }

            // 保留当前页码
            if (inspirationCurrentPage.value > 1) {
                inspirationQuery.page = inspirationCurrentPage.value.toString();
            }

            router.push({ path: '/inspirations', query: inspirationQuery });
            // 加载灵感数据
            loadInspirationData();
        };

        const switchToLoginView = () => {
            router.push('/login');

        };

        // 日期格式化函数
        const formatDate = (date) => {
            if (!date) return '';
            const d = new Date(date);
            return d.toLocaleDateString('zh-CN', {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit'
            });
        };

        // 灵感广场相关方法
        const loadInspirationData = async (forceRefresh = false) => {
            try {
                // 如果不是强制刷新，先尝试从缓存加载
                // 构建缓存键，包含分页和过滤条件
                const cacheKey = `${TEMPLATES_CACHE_KEY}_${inspirationCurrentPage.value}_${inspirationPageSize.value}_${selectedInspirationCategory.value}_${inspirationSearchQuery.value}`;

                if (!forceRefresh) {
                const cachedData = loadFromCache(cacheKey, TEMPLATES_CACHE_EXPIRY);
                if (cachedData && cachedData.templates) {
                    console.log(`成功从缓存加载灵感模板数据${cacheKey}:`, cachedData.templates);
                    inspirationItems.value = cachedData.templates;
                    InspirationCategories.value = cachedData.all_categories;
                        // 如果有分页信息也加载
                        if (cachedData.pagination) {
                            inspirationPagination.value = cachedData.pagination;
                        }
                    preloadTemplateFilesUrl(inspirationItems.value);
                    return;
                    }
                }

                // 缓存中没有或强制刷新，从API加载
                const params = new URLSearchParams();
                if (selectedInspirationCategory.value) {
                    params.append('category', selectedInspirationCategory.value);
                }
                if (inspirationSearchQuery.value) {
                    params.append('search', inspirationSearchQuery.value);
                }
                if (inspirationCurrentPage.value) {
                    params.append('page', inspirationCurrentPage.value.toString());
                }
                if (inspirationPageSize.value) {
                    params.append('page_size', inspirationPageSize.value.toString());
                }

                const apiUrl = `/api/v1/template/tasks${params.toString() ? '?' + params.toString() : ''}`;
                const response = await publicApiCall(apiUrl);
                if (response.ok) {
                    const data = await response.json();
                    inspirationItems.value = data.templates || [];
                    InspirationCategories.value = data.categories || [];
                    inspirationPagination.value = data.pagination || null;

                    // 缓存模板数据
                    saveToCache(cacheKey, {
                        templates: inspirationItems.value,
                        pagination: inspirationPagination.value,
                        all_categories: InspirationCategories.value,
                        category: selectedInspirationCategory.value,
                        search: inspirationSearchQuery.value,
                        page: inspirationCurrentPage.value,
                        page_size: inspirationPageSize.value,
                    });

                    console.log('缓存灵感模板数据成功:', inspirationItems.value.length, '个模板');
                    // 强制触发响应式更新
                    await nextTick();

                    // 强制刷新分页组件
                    inspirationPaginationKey.value++;

                    // 使用新的模板文件预加载逻辑
                    preloadTemplateFilesUrl(inspirationItems.value);
                } else {
                    console.warn('加载模板数据失败');
                }
            } catch (error) {
                console.warn('加载模板数据失败:', error);
            }
        };


        // 选择分类
        const selectInspirationCategory = async (category) => {
            isPageLoading.value = true;
            // 如果点击的是当前分类，不重复请求
            if (selectedInspirationCategory.value === category) {
                isPageLoading.value = false;
                return;
            }

            // 更新分类
            selectedInspirationCategory.value = category;

            // 重置页码为1
            inspirationCurrentPage.value = 1;
            inspirationPageInput.value = 1;

            // 清空当前数据，显示加载状态
            inspirationItems.value = [];
            inspirationPagination.value = null;

            // 重新加载数据
            await loadInspirationData(); // 强制刷新，不使用缓存
            isPageLoading.value = false;
        };

        // 搜索防抖定时器
        let searchTimeout = null;

        // 处理搜索
        const handleInspirationSearch = async () => {
            isLoading.value = true;
            // 清除之前的定时器
            if (searchTimeout) {
                clearTimeout(searchTimeout);
            }

            // 设置防抖延迟
            searchTimeout = setTimeout(async () => {
                // 重置页码为1
                inspirationCurrentPage.value = 1;
                inspirationPageInput.value = 1;

                // 清空当前数据，显示加载状态
                inspirationItems.value = [];
                inspirationPagination.value = null;

                // 重新加载数据
                await loadInspirationData(); // 强制刷新，不使用缓存
                isPageLoading.value = false;
            }, 500); // 500ms 防抖延迟
        };

        // 全局视频播放管理
        let currentPlayingVideo = null;
        let currentLoadingVideo = null; // 跟踪正在等待加载的视频

        // 更新视频播放按钮图标
        const updateVideoIcon = (video, isPlaying) => {
            // 查找视频容器中的播放按钮
            const container = video.closest('.relative');
            if (!container) return;

            // 查找移动端播放按钮
            const playButton = container.querySelector('button[class*="absolute"][class*="bottom-3"]');
            if (playButton) {
                const icon = playButton.querySelector('i');
                if (icon) {
                    icon.className = isPlaying ? 'fas fa-pause text-sm' : 'fas fa-play text-sm';
                }
            }
        };

        // 处理视频播放结束
        const onVideoEnded = (event) => {
            const video = event.target;
            console.log('视频播放完毕:', video.src);

            // 重置视频到开始位置
            video.currentTime = 0;

            // 更新播放按钮图标为播放状态
            updateVideoIcon(video, false);

            // 如果播放完毕的是当前播放的视频，清除引用
            if (currentPlayingVideo === video) {
                currentPlayingVideo = null;
                console.log('当前播放视频播放完毕');
            }
        };

        // 视频播放控制
        const playVideo = (event) => {
            const video = event.target;

            // 检查视频是否已加载完成
            if (video.readyState < 2) { // HAVE_CURRENT_DATA
                console.log('视频还没加载完成，忽略鼠标悬停播放');
                return;
            }

            // 如果当前有视频在播放，先暂停它
            if (currentPlayingVideo && currentPlayingVideo !== video) {
                currentPlayingVideo.pause();
                currentPlayingVideo.currentTime = 0;
                // 更新上一个视频的图标
                updateVideoIcon(currentPlayingVideo, false);
                console.log('暂停上一个视频');
            }

            // 视频已加载完成，可以播放
            video.currentTime = 0; // 从头开始播放
            video.play().then(() => {
                // 播放成功，更新当前播放视频
                currentPlayingVideo = video;
                console.log('开始播放新视频');
            }).catch(e => {
                console.log('视频播放失败:', e);
                currentPlayingVideo = null;
                video.pause();
                video.currentTime = 0;
            });
        };

        const pauseVideo = (event) => {
            const video = event.target;

            // 检查视频是否已加载完成
            if (video.readyState < 2) { // HAVE_CURRENT_DATA
                console.log('视频还没加载完成，忽略鼠标离开暂停');
                return;
            }

            video.pause();
            video.currentTime = 0;

            // 更新视频图标
            updateVideoIcon(video, false);

            // 如果暂停的是当前播放的视频，清除引用
            if (currentPlayingVideo === video) {
                currentPlayingVideo = null;
                console.log('暂停当前播放视频');
            }
        };

        // 移动端视频播放切换
        const toggleVideoPlay = (event) => {
            const button = event.target.closest('button');
            if (!button) {
                console.error('toggleVideoPlay: 未找到按钮元素');
                return;
            }

            const video = button.parentElement.querySelector('video');
            if (!video) {
                console.error('toggleVideoPlay: 未找到视频元素');
                return;
            }

            const icon = button.querySelector('i');

            if (video.paused) {
                // 如果当前有视频在播放，先暂停它
                if (currentPlayingVideo && currentPlayingVideo !== video) {
                    currentPlayingVideo.pause();
                    currentPlayingVideo.currentTime = 0;
                    // 更新上一个视频的图标
                    updateVideoIcon(currentPlayingVideo, false);
                    console.log('暂停上一个视频（移动端）');
                }

                // 如果当前有视频在等待加载，取消它的等待状态
                if (currentLoadingVideo && currentLoadingVideo !== video) {
                    currentLoadingVideo = null;
                    console.log('取消上一个视频的加载等待（移动端）');
                }

                // 检查视频是否已加载完成
                if (video.readyState >= 2) { // HAVE_CURRENT_DATA
                    // 视频已加载完成，直接播放
                    video.currentTime = 0;
                    video.play().then(() => {
                        icon.className = 'fas fa-pause text-sm';
                        currentPlayingVideo = video;
                        console.log('开始播放新视频（移动端）');
                    }).catch(e => {
                        console.log('视频播放失败:', e);
                        icon.className = 'fas fa-play text-sm';
                        currentPlayingVideo = null;
                    });
                } else {
                    // 视频未加载完成，显示loading并等待
                    console.log('视频还没加载完成，等待加载（移动端）, readyState:', video.readyState);
                    icon.className = 'fas fa-spinner fa-spin text-sm';
                    currentLoadingVideo = video;

                    // 主动触发视频加载
                    video.load();

                    // 设置超时保护（10秒后如果还未加载完成，重置状态）
                    const loadingTimeout = setTimeout(() => {
                        if (currentLoadingVideo === video) {
                            console.warn('视频加载超时（移动端）');
                            icon.className = 'fas fa-play text-sm';
                            currentLoadingVideo = null;
                            showAlert(t('videoLoadTimeout'), 'warning');
                        }
                    }, 10000);

                    // 等待视频可以播放
                    const playHandler = () => {
                        clearTimeout(loadingTimeout);

                        // 检查这个视频是否仍然是当前等待加载的视频
                        if (currentLoadingVideo === video) {
                            currentLoadingVideo = null;
                            video.currentTime = 0;
                            video.play().then(() => {
                                icon.className = 'fas fa-pause text-sm';
                                currentPlayingVideo = video;
                                console.log('开始播放新视频（移动端-延迟加载）');
                            }).catch(e => {
                                console.log('视频播放失败:', e);
                                icon.className = 'fas fa-play text-sm';
                                currentPlayingVideo = null;
                            });
                        } else {
                            // 这个视频的加载等待已被取消，重置图标
                            icon.className = 'fas fa-play text-sm';
                            console.log('视频加载完成但等待已被取消（移动端）');
                        }

                        // 移除事件监听器
                        video.removeEventListener('canplay', playHandler);
                        video.removeEventListener('error', errorHandler);
                    };

                    const errorHandler = () => {
                        clearTimeout(loadingTimeout);
                        console.error('视频加载失败（移动端）');
                        icon.className = 'fas fa-play text-sm';
                        currentLoadingVideo = null;

                        // 移除事件监听器
                        video.removeEventListener('canplay', playHandler);
                        video.removeEventListener('error', errorHandler);
                    };

                    // 使用 canplay 事件，比 loadeddata 更适合移动端
                    video.addEventListener('canplay', playHandler, { once: true });
                    video.addEventListener('error', errorHandler, { once: true });
                }
            } else {
                video.pause();
                video.currentTime = 0;
                icon.className = 'fas fa-play text-sm';

                // 如果暂停的是当前播放的视频，清除引用
                if (currentPlayingVideo === video) {
                    currentPlayingVideo = null;
                    console.log('暂停当前播放视频（移动端）');
                }

                // 如果暂停的是当前等待加载的视频，清除引用
                if (currentLoadingVideo === video) {
                    currentLoadingVideo = null;
                    console.log('取消当前等待加载的视频（移动端）');
                }
            }
        };

        // 暂停所有视频
        const pauseAllVideos = () => {
            if (currentPlayingVideo) {
                currentPlayingVideo.pause();
                currentPlayingVideo.currentTime = 0;
                // 更新视频图标
                updateVideoIcon(currentPlayingVideo, false);
                currentPlayingVideo = null;
                console.log('暂停所有视频');
            }

            // 清理等待加载的视频状态
            if (currentLoadingVideo) {
                // 重置等待加载的视频图标
                const loadingContainer = currentLoadingVideo.closest('.relative');
                if (loadingContainer) {
                    const loadingButton = loadingContainer.querySelector('button[class*="absolute"][class*="bottom-3"]');
                    if (loadingButton) {
                        const loadingIcon = loadingButton.querySelector('i');
                        if (loadingIcon) {
                            loadingIcon.className = 'fas fa-play text-sm';
                        }
                    }
                }
                currentLoadingVideo = null;
                console.log('取消所有等待加载的视频');
            }
        };

        const onVideoLoaded = (event) => {
            const video = event.target;
            // 视频加载完成，准备播放
            console.log('视频加载完成:', video.src);

            // 更新视频加载状态（使用视频的实际src）
            setVideoLoaded(video.src, true);

            // 触发Vue的响应式更新
            videoLoadedStates.value = new Map(videoLoadedStates.value);
        };

        const onVideoError = (event) => {
            const video = event.target;
            console.error('视频加载失败:', video.src, event);
            const img = event.target;
            const parent = img.parentElement;
            parent.innerHTML = '<div class="w-full h-44 bg-laser-purple/20 flex items-center justify-center"><i class="fas fa-video text-gradient-icon text-xl"></i></div>';
            // 回退到图片
        };

        // 预览模板详情
        const previewTemplateDetail = (item, updateRoute = true) => {
            selectedTemplate.value = item;
            showTemplateDetailModal.value = true;

            // 只在需要时更新路由到模板详情页面
            if (updateRoute && item?.task_id) {
                router.push(`/template/${item.task_id}`);
            }
        };

        // 关闭模板详情弹窗
        const closeTemplateDetailModal = () => {
            showTemplateDetailModal.value = false;
            selectedTemplate.value = null;
            // 移除自动路由跳转，让调用方决定路由行为
        };

        // 显示图片放大
        const showImageZoom = (imageUrl) => {
            zoomedImageUrl.value = imageUrl;
            showImageZoomModal.value = true;
        };

        // 关闭图片放大弹窗
        const closeImageZoomModal = () => {
            showImageZoomModal.value = false;
            zoomedImageUrl.value = '';
        };

        // 通过后端API代理获取文件（避免CORS问题）
        const fetchFileThroughProxy = async (fileKey, fileType) => {
            try {
                // 尝试通过后端API代理获取文件
                const proxyUrl = `/api/v1/template/asset/${fileType}/${fileKey}`;
                const response = await apiRequest(proxyUrl);

                if (response && response.ok) {
                    return await response.blob();
                }

                // 如果代理API不存在，尝试直接获取URL然后fetch
                const fileUrl = await getTemplateFileUrlAsync(fileKey, fileType);
                if (!fileUrl) {
                    return null;
                }

                // 检查是否是同源URL
                const urlObj = new URL(fileUrl, window.location.origin);
                const isSameOrigin = urlObj.origin === window.location.origin;

                if (isSameOrigin) {
                    // 同源，直接fetch
                    const directResponse = await fetch(fileUrl);
                    if (directResponse.ok) {
                        return await directResponse.blob();
                    }
                } else {
                    // 跨域，尝试使用no-cors模式（但这样无法读取响应）
                    // 或者使用img/audio元素加载（不适用于需要File对象的情况）
                    // 这里我们尝试直接fetch，如果失败会抛出错误
                    try {
                        const directResponse = await fetch(fileUrl, { mode: 'cors' });
                        if (directResponse.ok) {
                            return await directResponse.blob();
                        }
                    } catch (corsError) {
                        console.warn('CORS错误，尝试使用代理:', corsError);
                        // 如果后端有代理API，应该使用上面的代理方式
                        // 如果没有，这里会返回null，然后调用方会显示错误
                    }
                }

                return null;
            } catch (error) {
                console.error('获取文件失败:', error);
                return null;
            }
        };

        // 应用模板图片
        const applyTemplateImage = async (template) => {
            if (!template?.inputs?.input_image) {
                showAlert(t('applyImageFailed'), 'danger');
                return;
            }

            try {
                // 先设置任务类型（如果模板有任务类型）
                if (template.task_type && (template.task_type === 'i2v' || template.task_type === 's2v')) {
                    selectedTaskId.value = template.task_type;
                }

                // 检查当前任务类型是否支持图片
                if (selectedTaskId.value !== 'i2v' && selectedTaskId.value !== 's2v') {
                    showAlert(t('applyImageFailed'), 'danger');
                    return;
                }

                // 获取图片URL（用于预览）
                const imageUrl = await getTemplateFileUrlAsync(template.inputs.input_image, 'images');
                if (!imageUrl) {
                    console.error('无法获取模板图片URL:', template.inputs.input_image);
                    showAlert(t('applyImageFailed'), 'danger');
                    return;
                }

                // 根据任务类型设置图片
                const currentForm = getCurrentForm();
                if (currentForm) {
                    currentForm.imageUrl = imageUrl;
                    // Reset detected faces
                    if (selectedTaskId.value === 'i2v') {
                        i2vForm.value.detectedFaces = [];
                    } else if (selectedTaskId.value === 's2v') {
                        s2vForm.value.detectedFaces = [];
                    }
                }

                // 设置预览
                setCurrentImagePreview(imageUrl);

                // 加载图片文件（与useTemplate相同的逻辑）
                try {
                    // 直接使用获取到的URL fetch（与useTemplate相同）
                    const imageResponse = await fetch(imageUrl);
                    if (imageResponse.ok) {
                        const blob = await imageResponse.blob();
                        // 验证返回的是图片而不是HTML
                        if (blob.type && blob.type.startsWith('text/html')) {
                            console.error('返回的是HTML而不是图片:', blob.type);
                            showAlert(t('applyImageFailed'), 'danger');
                            return;
                        }
                        const filename = template.inputs.input_image || 'template_image.jpg';
                        const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });
                        if (currentForm) {
                            currentForm.imageFile = file;
                        }
                        console.log('模板图片文件已加载');

                        // 不再自动检测人脸，等待用户手动打开多角色模式开关
                    } else {
                        console.warn('Failed to fetch image from URL:', imageUrl);
                        showAlert(t('applyImageFailed'), 'danger');
                        return;
                    }
                } catch (error) {
                    console.error('Failed to load template image file:', error);
                    showAlert(t('applyImageFailed'), 'danger');
                    return;
                }
                updateUploadedContentStatus();

                // 关闭所有弹窗的辅助函数
                const closeAllModals = () => {
                    closeTaskDetailModal(); // 使用函数确保状态完全重置
                    showVoiceTTSModal.value = false;
                    closeTemplateDetailModal(); // 使用函数确保状态完全重置
                    showImageTemplates.value = false;
                    showAudioTemplates.value = false;
                    showPromptModal.value = false;
                    closeImageZoomModal(); // 使用函数确保状态完全重置
                };

                // 跳转到创作区域的函数
                const scrollToCreationArea = () => {
                    // 先关闭所有弹窗
                    closeAllModals();

                    // 如果不在生成页面，先切换视图
                    if (router.currentRoute.value.path !== '/generate') {
                        switchToCreateView();
                        // 等待路由切换完成后再展开和滚动
                        setTimeout(() => {
                            expandCreationArea();
                            setTimeout(() => {
                                // 滚动到顶部（TopBar 之后的位置，约60px）
                                const mainScrollable = document.querySelector('.main-scrollbar');
                                if (mainScrollable) {
                                    mainScrollable.scrollTo({
                                        top: 0,
                                        behavior: 'smooth'
                                    });
                                }
                            }, 100);
                        }, 100);
                    } else {
                        // 已经在生成页面，直接展开和滚动
                        expandCreationArea();
                        setTimeout(() => {
                            // 滚动到顶部（TopBar 之后的位置，约60px）
                            const mainScrollable = document.querySelector('.main-scrollbar');
                            if (mainScrollable) {
                                mainScrollable.scrollTo({
                                    top: 0,
                                    behavior: 'smooth'
                                });
                            }
                        }, 100);
                    }
                };

                showAlert(t('imageApplied'), 'success', {
                    label: t('view'),
                    onClick: scrollToCreationArea
                });
            } catch (error) {
                console.error('应用图片失败:', error);
                showAlert(t('applyImageFailed'), 'danger');
            }
        };

        // 应用模板音频
        const applyTemplateAudio = async (template) => {
            if (!template?.inputs?.input_audio) {
                showAlert(t('applyAudioFailed'), 'danger');
                return;
            }

            try {
                // 先设置任务类型（如果模板有任务类型）
                if (template.task_type && template.task_type === 's2v') {
                    selectedTaskId.value = template.task_type;
                }

                // 检查当前任务类型是否支持音频
                if (selectedTaskId.value !== 's2v') {
                    showAlert(t('applyAudioFailed'), 'danger');
                    return;
                }

                // 获取音频URL（用于预览）
                const audioUrl = await getTemplateFileUrlAsync(template.inputs.input_audio, 'audios');
                if (!audioUrl) {
                    console.error('无法获取模板音频URL:', template.inputs.input_audio);
                    showAlert(t('applyAudioFailed'), 'danger');
                    return;
                }

                // 设置音频文件
                const currentForm = getCurrentForm();
                if (currentForm) {
                    currentForm.audioUrl = audioUrl;
                }

                // 设置预览
                setCurrentAudioPreview(audioUrl);

                // 加载音频文件（与useTemplate相同的逻辑）
                try {
                    // 直接使用获取到的URL fetch（与useTemplate相同）
                    const audioResponse = await fetch(audioUrl);
                    if (audioResponse.ok) {
                        const blob = await audioResponse.blob();
                        // 验证返回的是音频而不是HTML
                        if (blob.type && blob.type.startsWith('text/html')) {
                            console.error('返回的是HTML而不是音频:', blob.type);
                            showAlert(t('applyAudioFailed'), 'danger');
                            return;
                        }
                        const filename = template.inputs.input_audio || 'template_audio.mp3';

                        // 根据文件扩展名确定正确的MIME类型
                        let mimeType = blob.type;
                        if (!mimeType || mimeType === 'application/octet-stream') {
                            const ext = filename.toLowerCase().split('.').pop();
                            const mimeTypes = {
                                'mp3': 'audio/mpeg',
                                'wav': 'audio/wav',
                                'mp4': 'audio/mp4',
                                'aac': 'audio/aac',
                                'ogg': 'audio/ogg',
                                'm4a': 'audio/mp4'
                            };
                            mimeType = mimeTypes[ext] || 'audio/mpeg';
                        }

                        const file = new File([blob], filename, { type: mimeType });
                        if (currentForm) {
                            currentForm.audioFile = file;
                        }
                        console.log('模板音频文件已加载');
                    } else {
                        console.warn('Failed to fetch audio from URL:', audioUrl);
                        showAlert(t('applyAudioFailed'), 'danger');
                        return;
                    }
                } catch (error) {
                    console.error('Failed to load template audio file:', error);
                    showAlert(t('applyAudioFailed'), 'danger');
                    return;
                }
                        updateUploadedContentStatus();

                // 关闭所有弹窗的辅助函数
                const closeAllModals = () => {
                    closeTaskDetailModal(); // 使用函数确保状态完全重置
                    showVoiceTTSModal.value = false;
                    closeTemplateDetailModal(); // 使用函数确保状态完全重置
                    showImageTemplates.value = false;
                    showAudioTemplates.value = false;
                    showPromptModal.value = false;
                    closeImageZoomModal(); // 使用函数确保状态完全重置
                };

                // 跳转到创作区域的函数
                const scrollToCreationArea = () => {
                    // 先关闭所有弹窗
                    closeAllModals();

                    // 如果不在生成页面，先切换视图
                    if (router.currentRoute.value.path !== '/generate') {
                        switchToCreateView();
                        // 等待路由切换完成后再展开和滚动
                        setTimeout(() => {
                            expandCreationArea();
                            setTimeout(() => {
                                // 滚动到顶部（TopBar 之后的位置，约60px）
                                const mainScrollable = document.querySelector('.main-scrollbar');
                                if (mainScrollable) {
                                    mainScrollable.scrollTo({
                                        top: 0,
                                        behavior: 'smooth'
                                    });
                                }
                            }, 100);
                        }, 100);
                    } else {
                        // 已经在生成页面，直接展开和滚动
                        expandCreationArea();
                        setTimeout(() => {
                            // 滚动到顶部（TopBar 之后的位置，约60px）
                            const mainScrollable = document.querySelector('.main-scrollbar');
                            if (mainScrollable) {
                                mainScrollable.scrollTo({
                                    top: 0,
                                    behavior: 'smooth'
                                });
                            }
                        }, 100);
                    }
                };

                showAlert(t('audioApplied'), 'success', {
                    label: t('view'),
                    onClick: scrollToCreationArea
                });
            } catch (error) {
                        console.error('应用音频失败:', error);
                        showAlert(t('applyAudioFailed'), 'danger');
            }
        };

        // 应用模板Prompt
        const applyTemplatePrompt = (template) => {
            if (template?.params?.prompt) {
                const currentForm = getCurrentForm();
                if (currentForm) {
                    currentForm.prompt = template.params.prompt;
                    updateUploadedContentStatus();
                    showAlert(t('promptApplied'), 'success');
                }
            }
        };

        // 复制文本到剪贴板的辅助函数（支持移动端降级）
        const copyToClipboard = async (text) => {
            // 检查是否支持现代 Clipboard API
            if (navigator.clipboard && navigator.clipboard.writeText) {
                try {
                    await navigator.clipboard.writeText(text);
                    return true;
            } catch (error) {
                    console.warn('Clipboard API 失败，尝试降级方案:', error);
                    // 降级到传统方法
                }
            }

            // 降级方案：使用传统方法（适用于移动端和不支持Clipboard API的浏览器）
            try {
                const textArea = document.createElement('textarea');
                textArea.value = text;

                // 移动端需要元素可见且可聚焦，所以先设置可见样式
                textArea.style.position = 'fixed';
                textArea.style.left = '0';
                textArea.style.top = '0';
                textArea.style.width = '2em';
                textArea.style.height = '2em';
                textArea.style.padding = '0';
                textArea.style.border = 'none';
                textArea.style.outline = 'none';
                textArea.style.boxShadow = 'none';
                textArea.style.background = 'transparent';
                textArea.style.opacity = '0';
                textArea.style.zIndex = '-1';
                textArea.setAttribute('readonly', '');
                textArea.setAttribute('aria-hidden', 'true');
                textArea.setAttribute('tabindex', '-1');

                document.body.appendChild(textArea);

                // 聚焦元素（移动端需要）
                textArea.focus();
                textArea.select();

                // 移动端需要 setSelectionRange
                if (textArea.setSelectionRange) {
                    textArea.setSelectionRange(0, text.length);
                }

                // 尝试复制
                let successful = false;
                try {
                    successful = document.execCommand('copy');
                } catch (e) {
                    console.warn('execCommand 执行失败:', e);
                }

                // 立即移除元素
                document.body.removeChild(textArea);

                if (successful) {
                    return true;
                } else {
                    // 如果仍然失败，尝试另一种方法：在视口中心创建可见的输入框
                    return await fallbackCopyToClipboard(text);
                }
            } catch (error) {
                console.error('复制失败，尝试备用方案:', error);
                // 尝试备用方案
                return await fallbackCopyToClipboard(text);
            }
        };

        // 备用复制方案：显示一个可选择的文本区域（Apple风格）
        const fallbackCopyToClipboard = async (text) => {
            return new Promise((resolve) => {
                // 创建遮罩层
                const overlay = document.createElement('div');
                overlay.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0, 0, 0, 0.5);
                    backdrop-filter: blur(8px);
                    -webkit-backdrop-filter: blur(8px);
                    z-index: 10000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                `;

                // 创建弹窗容器（Apple风格）
                const container = document.createElement('div');
                container.style.cssText = `
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(20px) saturate(180%);
                    -webkit-backdrop-filter: blur(20px) saturate(180%);
                    border-radius: 20px;
                    padding: 24px;
                    max-width: 90%;
                    width: 100%;
                    max-width: 500px;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                `;

                // 深色模式支持
                if (document.documentElement.classList.contains('dark')) {
                    container.style.background = 'rgba(30, 30, 30, 0.95)';
                }

                const title = document.createElement('div');
                title.textContent = t('copyLink') || '复制链接';
                title.style.cssText = `
                    font-size: 18px;
                    font-weight: 600;
                    color: #1d1d1f;
                    margin-bottom: 12px;
                    text-align: center;
                `;
                if (document.documentElement.classList.contains('dark')) {
                    title.style.color = '#f5f5f7';
                }

                const message = document.createElement('div');
                message.textContent = t('pleaseCopyManually') || '请手动选择并复制下面的文本';
                message.style.cssText = `
                    color: #86868b;
                    font-size: 14px;
                    margin-bottom: 16px;
                    text-align: center;
                `;
                if (document.documentElement.classList.contains('dark')) {
                    message.style.color = '#98989d';
                }

                const input = document.createElement('input');
                input.type = 'text';
                input.value = text;
                input.readOnly = true;
                input.style.cssText = `
                    width: 100%;
                    padding: 12px 16px;
                    font-size: 14px;
                    border: 1px solid rgba(0, 0, 0, 0.1);
                    border-radius: 12px;
                    background: rgba(255, 255, 255, 0.8);
                    color: #1d1d1f;
                    margin-bottom: 16px;
                    box-sizing: border-box;
                    -webkit-appearance: none;
                    appearance: none;
                `;
                if (document.documentElement.classList.contains('dark')) {
                    input.style.border = '1px solid rgba(255, 255, 255, 0.1)';
                    input.style.background = 'rgba(44, 44, 46, 0.8)';
                    input.style.color = '#f5f5f7';
                }

                const button = document.createElement('button');
                button.textContent = t('close') || '关闭';
                button.style.cssText = `
                    width: 100%;
                    padding: 12px 24px;
                    background: var(--brand-primary, #007AFF);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    cursor: pointer;
                    font-size: 15px;
                    font-weight: 600;
                    transition: all 0.2s;
                `;
                button.onmouseover = () => {
                    button.style.opacity = '0.9';
                    button.style.transform = 'scale(1.02)';
                };
                button.onmouseout = () => {
                    button.style.opacity = '1';
                    button.style.transform = 'scale(1)';
                };

                container.appendChild(title);
                container.appendChild(message);
                container.appendChild(input);
                container.appendChild(button);
                overlay.appendChild(container);

                const close = () => {
                    document.body.removeChild(overlay);
                    resolve(false); // 返回false表示需要用户手动复制
                };

                button.onclick = close;
                overlay.onclick = (e) => {
                    if (e.target === overlay) close();
                };

                document.body.appendChild(overlay);

                // 选中文本（延迟以确保DOM已渲染）
                setTimeout(() => {
                    input.focus();
                    input.select();
                    if (input.setSelectionRange) {
                        input.setSelectionRange(0, text.length);
                    }
                }, 150);
            });
        };

        // 复制Prompt到剪贴板
        const copyPrompt = async (promptText) => {
            if (!promptText) return;

            try {
                // 使用辅助函数复制，支持移动端
                const success = await copyToClipboard(promptText);
                if (success) {
                showAlert(t('promptCopied'), 'success');
                }
                // 如果返回false，说明已经显示了手动复制的弹窗，不需要额外提示
            } catch (error) {
                console.error('复制Prompt失败:', error);
                showAlert(t('copyFailed'), 'error');
            }
        };

        // 使用模板
        const useTemplate = async (item) => {
            if (!item) {
                showAlert(t('templateDataIncomplete'), 'danger');
                return;
            }
            console.log('使用模板:', item);

            try {
                // 开始模板加载
                templateLoading.value = true;
                templateLoadingMessage.value = t('prefillLoadingTemplate');

                // 先设置任务类型
                selectedTaskId.value = item.task_type;

                // 获取当前表单
                const currentForm = getCurrentForm();

                // 设置表单数据
                currentForm.prompt = item.params?.prompt || '';
                currentForm.negative_prompt = item.params?.negative_prompt || '';
                currentForm.seed = item.params?.seed || 42;
                currentForm.model_cls = item.model_cls || '';
                currentForm.stage = item.stage || '';

                // 立即关闭模板详情并切换到创建视图，后续资源异步加载
                showTemplateDetailModal.value = false;
                selectedTemplate.value = null;
                isCreationAreaExpanded.value = true;
                switchToCreateView();

                // 创建加载Promise数组
                const loadingPromises = [];

                // 如果有输入图片，先获取正确的URL，然后加载文件
                if (item.inputs && item.inputs.input_image) {
                    // 异步获取图片URL
                    const imageLoadPromise = new Promise(async (resolve) => {
                        try {
                            // 先获取正确的URL
                            const imageUrl = await getTemplateFileUrlAsync(item.inputs.input_image, 'images');
                            if (!imageUrl) {
                                console.warn('无法获取模板图片URL:', item.inputs.input_image);
                                resolve();
                                return;
                            }

                            currentForm.imageUrl = imageUrl;
                            setCurrentImagePreview(imageUrl); // 设置正确的URL作为预览
                            console.log('模板输入图片URL:', imageUrl);

                            // Reset detected faces
                            if (selectedTaskId.value === 'i2v') {
                                i2vForm.value.detectedFaces = [];
                            } else if (selectedTaskId.value === 's2v') {
                                s2vForm.value.detectedFaces = [];
                            }

                            // 加载图片文件
                            const imageResponse = await fetch(imageUrl);
                            if (imageResponse.ok) {
                                const blob = await imageResponse.blob();
                                const filename = item.inputs.input_image;
                                const file = new File([blob], filename, { type: blob.type });
                                currentForm.imageFile = file;
                                console.log('模板图片文件已加载');

                                // 不再自动检测人脸，等待用户手动打开多角色模式开关
                            } else {
                                console.warn('Failed to fetch image from URL:', imageUrl);
                            }
                        } catch (error) {
                            console.warn('Failed to load template image file:', error);
                        }
                        resolve();
                    });
                    loadingPromises.push(imageLoadPromise);
                }

                // 如果有输入音频，先获取正确的URL，然后加载文件
                if (item.inputs && item.inputs.input_audio) {
                    // 异步获取音频URL
                    const audioLoadPromise = new Promise(async (resolve) => {
                        try {
                            // 先获取正确的URL
                            const audioUrl = await getTemplateFileUrlAsync(item.inputs.input_audio, 'audios');
                            if (!audioUrl) {
                                console.warn('无法获取模板音频URL:', item.inputs.input_audio);
                                resolve();
                                return;
                            }

                            currentForm.audioUrl = audioUrl;
                            setCurrentAudioPreview(audioUrl); // 设置正确的URL作为预览
                            console.log('模板输入音频URL:', audioUrl);

                            // 加载音频文件
                            const audioResponse = await fetch(audioUrl);
                            if (audioResponse.ok) {
                                const blob = await audioResponse.blob();
                                const filename = item.inputs.input_audio;

                                // 根据文件扩展名确定正确的MIME类型
                                let mimeType = blob.type;
                                if (!mimeType || mimeType === 'application/octet-stream') {
                                    const ext = filename.toLowerCase().split('.').pop();
                                    const mimeTypes = {
                                        'mp3': 'audio/mpeg',
                                        'wav': 'audio/wav',
                                        'mp4': 'audio/mp4',
                                        'aac': 'audio/aac',
                                        'ogg': 'audio/ogg',
                                        'm4a': 'audio/mp4'
                                    };
                                    mimeType = mimeTypes[ext] || 'audio/mpeg';
                                }

                                const file = new File([blob], filename, { type: mimeType });
                                currentForm.audioFile = file;
                                console.log('模板音频文件已加载');
                                // 使用FileReader生成data URL，与正常上传保持一致
                                const reader = new FileReader();
                                reader.onload = (e) => {
                                    setCurrentAudioPreview(e.target.result);
                                    console.log('模板音频预览已设置:', e.target.result.substring(0, 50) + '...');
                                };
                                reader.readAsDataURL(file);
                            } else {
                                console.warn('Failed to fetch audio from URL:', audioUrl);
                            }
                        } catch (error) {
                            console.warn('Failed to load template audio file:', error);
                        }
                        resolve();
                    });
                    loadingPromises.push(audioLoadPromise);
                }

                // 等待所有文件加载完成
                if (loadingPromises.length > 0) {
                    await Promise.all(loadingPromises);
                }

                showAlert(`模板加载完成`, 'success');
            } catch (error) {
                console.error('应用模板失败:', error);
                showAlert(`应用模板失败: ${error.message}`, 'danger');
            } finally {
                // 结束模板加载
                templateLoading.value = false;
                templateLoadingMessage.value = '';
            }
        };

        // 加载更多灵感
        const loadMoreInspiration = () => {
            showAlert(t('loadMoreInspirationComingSoon'), 'info');
        };

        // 新增：任务详情弹窗方法
        const openTaskDetailModal = (task) => {
            console.log('openTaskDetailModal called with task:', task);
            modalTask.value = task;
            showTaskDetailModal.value = true;
            // 只有不在 /generate 页面时才更新路由
            // 在 /generate 页面打开任务详情时，保持在当前页面
            const currentRoute = router.currentRoute.value;
            if (task?.task_id && currentRoute.path !== '/generate') {
                router.push(`/task/${task.task_id}`);
            }
        };

        const closeTaskDetailModal = () => {
            showTaskDetailModal.value = false;
            modalTask.value = null;
            // 只有当前路由是 /task/:id 时才跳转回 Projects
            // 如果在其他页面（如 /generate）打开的弹窗，关闭时保持在原页面
            const currentRoute = router.currentRoute.value;
            if (currentRoute.path.startsWith('/task/')) {
                // 从任务详情路由打开的，返回 Projects 页面
            router.push({ name: 'Projects' });
            }
            // 如果不是任务详情路由，不做任何路由跳转，保持在当前页面
        };

        // 新增：分享功能相关方法
        const generateShareUrl = (taskId) => {
            const baseUrl = window.location.origin;
            return `${baseUrl}/share/${taskId}`;
        };

        const copyShareLink = async (taskId, shareType = 'task') => {
            try {
                const token = localStorage.getItem('accessToken');
                if (!token) {
                    showAlert(t('pleaseLoginFirst'), 'warning');
                    return;
                }

                // 调用后端接口创建分享链接
                const response = await fetch('/api/v1/share/create', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${token}`
                    },
                    body: JSON.stringify({
                        task_id: taskId,
                        share_type: shareType
                    })
                });

                if (!response.ok) {
                    throw new Error('创建分享链接失败');
                }

                const data = await response.json();
                const shareUrl = `${window.location.origin}${data.share_url}`;

                // 使用辅助函数复制，支持移动端
                const success = await copyToClipboard(shareUrl);

                // 如果成功复制，显示成功提示
                if (success) {
                // 显示带操作按钮的alert
                showAlert(t('shareLinkCopied'), 'success', {
                    label: t('view'),
                    onClick: () => {
                        window.open(shareUrl, '_blank');
                    }
                });
                }
                // 如果返回false，说明已经显示了手动复制的弹窗，不需要额外提示
            } catch (err) {
                console.error('复制失败:', err);
                showAlert(t('copyFailed'), 'error');
            }
        };

        const shareToSocial = (taskId, platform) => {
            const shareUrl = generateShareUrl(taskId);
            const task = modalTask.value;
            const title = task?.params?.prompt || t('aiGeneratedVideo');
            const description = t('checkOutThisAIGeneratedVideo');

            let shareUrlWithParams = '';

            switch (platform) {
                case 'twitter':
                    shareUrlWithParams = `https://twitter.com/intent/tweet?text=${encodeURIComponent(title)}&url=${encodeURIComponent(shareUrl)}`;
                    break;
                case 'facebook':
                    shareUrlWithParams = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}`;
                    break;
                case 'linkedin':
                    shareUrlWithParams = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(shareUrl)}`;
                    break;
                case 'whatsapp':
                    shareUrlWithParams = `https://wa.me/?text=${encodeURIComponent(title + ' ' + shareUrl)}`;
                    break;
                case 'telegram':
                    shareUrlWithParams = `https://t.me/share/url?url=${encodeURIComponent(shareUrl)}&text=${encodeURIComponent(title)}`;
                    break;
                case 'weibo':
                    shareUrlWithParams = `https://service.weibo.com/share/share.php?url=${encodeURIComponent(shareUrl)}&title=${encodeURIComponent(title)}`;
                    break;
                default:
                    return;
            }

            window.open(shareUrlWithParams, '_blank', 'width=600,height=400');
        };

        // 新增：从路由参数打开任务详情
        const openTaskFromRoute = async (taskId) => {
            try {
                // 如果任务列表为空，先加载任务数据
                if (tasks.value.length === 0) {
                    await refreshTasks();
                }

                if (showTaskDetailModal.value && modalTask.value?.task_id === taskId) {
                    console.log('任务详情已打开，不重复打开');
                    return;
                }

                // 查找任务
                const task = tasks.value.find(t => t.task_id === taskId);
                if (task) {
                    modalTask.value = task;
                    openTaskDetailModal(task);
                } else {
                    // 如果任务不在当前列表中，尝试从API获取
                    showAlert(t('taskNotFound'), 'error');
                    router.push({ name: 'Projects' });
                }
            } catch (error) {
                console.error('打开任务失败:', error);
                showAlert(t('openTaskFailed'), 'error');
                router.push({ name: 'Projects' });
            }
        };

        // 新增：模板分享功能相关方法
        const generateTemplateShareUrl = (templateId) => {
            const baseUrl = window.location.origin;
            return `${baseUrl}/template/${templateId}`;
        };

        const copyTemplateShareLink = async (templateId) => {
            try {
                const shareUrl = generateTemplateShareUrl(templateId);
                // 使用辅助函数复制，支持移动端
                const success = await copyToClipboard(shareUrl);

                // 如果成功复制，显示成功提示
                if (success) {
                showAlert(t('templateShareLinkCopied'), 'success', {
                    label: t('view'),
                    onClick: () => {
                        window.open(shareUrl, '_blank');
                    }
                });
                }
                // 如果返回false，说明已经显示了手动复制的弹窗，不需要额外提示
            } catch (err) {
                console.error('复制模板分享链接失败:', err);
                showAlert(t('copyFailed'), 'error');
            }
        };

        const shareTemplateToSocial = (templateId, platform) => {
            const shareUrl = generateTemplateShareUrl(templateId);
            const template = selectedTemplate.value;
            const title = template?.params?.prompt || t('aiGeneratedTemplate');
            const description = t('checkOutThisAITemplate');

            let shareUrlWithParams = '';

            switch (platform) {
                case 'twitter':
                    shareUrlWithParams = `https://twitter.com/intent/tweet?text=${encodeURIComponent(title)}&url=${encodeURIComponent(shareUrl)}`;
                    break;
                case 'facebook':
                    shareUrlWithParams = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}`;
                    break;
                case 'linkedin':
                    shareUrlWithParams = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(shareUrl)}`;
                    break;
                case 'whatsapp':
                    shareUrlWithParams = `https://wa.me/?text=${encodeURIComponent(title + ' ' + shareUrl)}`;
                    break;
                case 'telegram':
                    shareUrlWithParams = `https://t.me/share/url?url=${encodeURIComponent(shareUrl)}&text=${encodeURIComponent(title)}`;
                    break;
                case 'weibo':
                    shareUrlWithParams = `https://service.weibo.com/share/share.php?url=${encodeURIComponent(shareUrl)}&title=${encodeURIComponent(title)}`;
                    break;
                default:
                    return;
            }

            window.open(shareUrlWithParams, '_blank', 'width=600,height=400');
        };

        // 新增：从路由参数打开模板详情
        const openTemplateFromRoute = async (templateId) => {
            try {
                // 如果模板列表为空，先加载模板数据
                if (inspirationItems.value.length === 0) {
                    await loadInspirationData();
                }

                if (showTemplateDetailModal.value && selectedTemplate.value?.task_id === templateId) {
                    console.log('模板详情已打开，不重复打开');
                    return;
                }

                // 查找模板
                const template = inspirationItems.value.find(t => t.task_id === templateId);
                if (template) {
                    selectedTemplate.value = template;
                    previewTemplateDetail(template);
                } else {
                    // 如果模板不在当前列表中，尝试从API获取
                    showAlert(t('templateNotFound'), 'error');
                    router.push({ name: 'Inspirations' });
                }
            } catch (error) {
                console.error('打开模板失败:', error);
                showAlert(t('openTemplateFailed'), 'error');
                router.push({ name: 'Inspirations' });
            }
        };

        // 精选模版相关数据
        const featuredTemplates = ref([]);
        const featuredTemplatesLoading = ref(false);

        // 主题管理
        const theme = ref('dark'); // 'light', 'dark' - 默认深色模式

        // 初始化主题
        const initTheme = () => {
            const savedTheme = localStorage.getItem('theme') || 'dark'; // 默认深色模式
            theme.value = savedTheme;
            applyTheme(savedTheme);
        };

        // 应用主题（优化版本，减少延迟）
        const applyTheme = (newTheme) => {
            const html = document.documentElement;

            // 使用 requestAnimationFrame 优化 DOM 操作
            requestAnimationFrame(() => {
                // 临时禁用过渡动画以提高切换速度
                html.classList.add('theme-transitioning');

                if (newTheme === 'dark') {
                    html.classList.add('dark');
                    html.style.colorScheme = 'dark';
                } else {
                    html.classList.remove('dark');
                    html.style.colorScheme = 'light';
                }

                // 短暂延迟后移除过渡禁用类，恢复平滑过渡
                setTimeout(() => {
                    html.classList.remove('theme-transitioning');
                }, 50);
            });
        };

        // 切换主题（优化版本）
        const toggleTheme = () => {
            const themes = ['light', 'dark'];
            const currentIndex = themes.indexOf(theme.value);
            const nextIndex = (currentIndex + 1) % themes.length;
            const nextTheme = themes[nextIndex];

            // 立即更新状态
            theme.value = nextTheme;

            // 异步保存到 localStorage，不阻塞 UI
            if (window.requestIdleCallback) {
                requestIdleCallback(() => {
                    localStorage.setItem('theme', nextTheme);
                }, { timeout: 100 });
            } else {
                // 回退方案：使用 setTimeout
                setTimeout(() => {
                    localStorage.setItem('theme', nextTheme);
                }, 0);
            }

            // 立即应用主题
            applyTheme(nextTheme);

            // 延迟显示提示，避免阻塞主题切换
            const themeNames = {
                'light': '浅色模式',
                'dark': '深色模式'
            };
            setTimeout(() => {
                showAlert(`已切换到${themeNames[nextTheme]}`, 'info');
            }, 100);
        };

        // 获取主题图标
        const getThemeIcon = () => {
            const iconMap = {
                'light': 'fas fa-sun',
                'dark': 'fas fa-moon'
            };
            return iconMap[theme.value] || 'fas fa-moon';
        };

        // 不需要认证的API调用（用于获取模版数据）
        const publicApiCall = async (endpoint, options = {}) => {
            const url = `${endpoint}`;
            const headers = {
                'Content-Type': 'application/json',
                ...options.headers
            };

            const response = await fetch(url, {
                ...options,
                headers
            });

            if (response.status === 400) {
                const error = await response.json();
                showAlert(error.message, 'danger');
                throw new Error(error.message);
            }

            // 添加50ms延迟，防止触发服务端频率限制
            await new Promise(resolve => setTimeout(resolve, 50));

            return response;
        };

        // 获取精选模版数据
        const loadFeaturedTemplates = async (forceRefresh = false) => {
            try {
                featuredTemplatesLoading.value = true;

                // 构建缓存键
                const cacheKey = `featured_templates_cache`;

                if (!forceRefresh) {
                    const cachedData = loadFromCache(cacheKey, TEMPLATES_CACHE_EXPIRY);
                    if (cachedData && cachedData.templates) {
                        console.log('从缓存加载精选模版数据:', cachedData.templates.length, '个');
                        featuredTemplates.value = cachedData.templates;
                        featuredTemplatesLoading.value = false;
                        return;
                    }
                }

                // 从API获取精选模版数据（不需要认证）
                const params = new URLSearchParams();
                params.append('category', '精选');
                params.append('page_size', '50'); // 获取更多数据用于随机选择

                const apiUrl = `/api/v1/template/tasks?${params.toString()}`;
                const response = await publicApiCall(apiUrl);

                if (response.ok) {
                    const data = await response.json();
                    const templates = data.templates || [];

                    // 缓存数据
                    saveToCache(cacheKey, {
                        templates: templates,
                        timestamp: Date.now()
                    });

                    featuredTemplates.value = templates;
                    console.log('成功加载精选模版数据:', templates.length, '个模版');
                } else {
                    console.warn('加载精选模版数据失败');
                    featuredTemplates.value = [];
                }
            } catch (error) {
                console.warn('加载精选模版数据失败:', error);
                featuredTemplates.value = [];
            } finally {
                featuredTemplatesLoading.value = false;
            }
        };

        // 获取随机精选模版
        const getRandomFeaturedTemplates = async (count = 10) => {
            try {
                featuredTemplatesLoading.value = true;

                // 如果当前没有数据，先加载
                if (featuredTemplates.value.length === 0) {
                    await loadFeaturedTemplates();
                }

                // 如果数据仍然为空，返回空数组
                if (featuredTemplates.value.length === 0) {
                    return [];
                }

                // 随机选择指定数量的模版
                const shuffled = [...featuredTemplates.value].sort(() => 0.5 - Math.random());
                const randomTemplates = shuffled.slice(0, count);

                return randomTemplates;
            } catch (error) {
                console.error('获取随机精选模版失败:', error);
                return [];
            } finally {
                featuredTemplatesLoading.value = false;
            }
        };
        const removeTtsHistoryEntry = (entryId) => {
            if (!entryId) return;
            const currentHistory = loadTtsHistory().filter(entry => entry.id !== entryId);
            saveTtsHistory(currentHistory);
        };

        const loadTtsHistory = () => {
            try {
                const stored = localStorage.getItem('ttsHistory');
                if (!stored) return [];
                const parsed = JSON.parse(stored);
                ttsHistory.value = Array.isArray(parsed) ? parsed : [];
                return ttsHistory.value;
            } catch (error) {
                console.error('加载TTS历史失败:', error);
                ttsHistory.value = [];
                return [];
            }
        };

        const saveTtsHistory = (historyList) => {
            try {
                localStorage.setItem('ttsHistory', JSON.stringify(historyList));
                ttsHistory.value = historyList;
            } catch (error) {
                console.error('保存TTS历史失败:', error);
            }
        };

        const addTtsHistoryEntry = (text = '', instruction = '') => {
            const trimmedText = (text || '').trim();
            const trimmedInstruction = (instruction || '').trim();

            if (!trimmedText && !trimmedInstruction) {
                return;
            }

            const currentHistory = loadTtsHistory();

            const existingIndex = currentHistory.findIndex(entry =>
                entry.text === trimmedText && entry.instruction === trimmedInstruction
            );

            const timestamp = new Date().toISOString();

            if (existingIndex !== -1) {
                const existingEntry = currentHistory.splice(existingIndex, 1)[0];
                existingEntry.timestamp = timestamp;
                currentHistory.unshift(existingEntry);
            } else {
                currentHistory.unshift({
                    id: Date.now(),
                    text: trimmedText,
                    instruction: trimmedInstruction,
                    timestamp
                });
            }

            if (currentHistory.length > 20) {
                currentHistory.length = 20;
            }

            saveTtsHistory(currentHistory);
        };

        const clearTtsHistory = () => {
            ttsHistory.value = [];
            localStorage.removeItem('ttsHistory');
        };

export {
            // 任务类型下拉菜单
            showTaskTypeMenu,
            showModelMenu,
            isLoggedIn,
            loading,
            loginLoading,
            initLoading,
            downloadLoading,
            downloadLoadingMessage,
            isLoading,
            isPageLoading,

            // 录音相关
            isRecording,
            recordingDuration,
            startRecording,
            stopRecording,
            formatRecordingDuration,

            loginWithGitHub,
            loginWithGoogle,
            // 短信登录相关
            phoneNumber,
            verifyCode,
            smsCountdown,
            showSmsForm,
            sendSmsCode,
            loginWithSms,
            handlePhoneEnter,
            handleVerifyCodeEnter,
            toggleSmsLogin,
            submitting,
            templateLoading,
            templateLoadingMessage,
            taskSearchQuery,
            currentUser,
            models,
            tasks,
            alert,
            showErrorDetails,
            showFailureDetails,
            confirmDialog,
            showConfirmDialog,
            showTaskDetailModal,
            modalTask,
            showVoiceTTSModal,
            showPodcastModal,
            currentTask,
            t2vForm,
            i2vForm,
            s2vForm,
            getCurrentForm,
            i2vImagePreview,
            s2vImagePreview,
            s2vAudioPreview,
            getCurrentImagePreview,
            getCurrentAudioPreview,
            getCurrentVideoPreview,
            setCurrentImagePreview,
            setCurrentAudioPreview,
            setCurrentVideoPreview,
            updateUploadedContentStatus,
            availableTaskTypes,
            availableModelClasses,
            currentTaskHints,
            currentHintIndex,
            startHintRotation,
            stopHintRotation,
            filteredTasks,
            selectedTaskId,
            selectedTask,
            selectedModel,
            selectedTaskFiles,
            loadingTaskFiles,
            statusFilter,
            pagination,
            paginationInfo,
            currentTaskPage,
            taskPageSize,
            taskPageInput,
            paginationKey,
            taskMenuVisible,
            toggleTaskMenu,
            closeAllTaskMenus,
            handleClickOutside,
            showAlert,
            setLoading,
            apiCall,
            logout,
            login,
            loadModels,
            sidebarCollapsed,
            sidebarWidth,
            showExpandHint,
            showGlow,
            isDefaultStateHidden,
            hideDefaultState,
            showDefaultState,
            isCreationAreaExpanded,
            hasUploadedContent,
            isContracting,
            expandCreationArea,
            contractCreationArea,
            taskFileCache,
            taskFileCacheLoaded,
            templateFileCache,
            templateFileCacheLoaded,
            loadTaskFiles,
            downloadFile,
            handleDownloadFile,
            viewFile,
            handleImageUpload,
            detectFacesInImage,
            faceDetecting,
            audioSeparating,
            cropFaceImage,
            updateFaceRoleName,
            toggleFaceEditing,
            saveFaceRoleName,
            selectTask,
            selectModel,
            resetForm,
            triggerImageUpload,
            triggerAudioUpload,
            removeImage,
            removeAudio,
            removeVideo,
            handleAudioUpload,
            handleVideoUpload,
            separateAudioTracks,
            updateSeparatedAudioRole,
            updateSeparatedAudioName,
            toggleSeparatedAudioEditing,
            saveSeparatedAudioName,
            loadImageAudioTemplates,
            selectImageTemplate,
            selectAudioTemplate,
            previewAudioTemplate,
            stopAudioPlayback,
            setAudioStopCallback,
            getTemplateFile,
            imageTemplates,
            audioTemplates,
            mergedTemplates,
            showImageTemplates,
            showAudioTemplates,
            mediaModalTab,
            templatePagination,
            templatePaginationInfo,
            templateCurrentPage,
            templatePageSize,
            templatePageInput,
            templatePaginationKey,
            imageHistory,
            audioHistory,
            showTemplates,
            showHistory,
            showPromptModal,
            promptModalTab,
            submitTask,
            fileToBase64,
            formatTime,
            refreshTasks,
            goToPage,
            jumpToPage,
            getVisiblePages,
            goToTemplatePage,
            jumpToTemplatePage,
            getVisibleTemplatePages,
            goToInspirationPage,
            jumpToInspirationPage,
            getVisibleInspirationPages,
            preloadTaskFilesUrl,
            preloadTemplateFilesUrl,
            loadTaskFilesFromCache,
            saveTaskFilesToCache,
            getTaskFileFromCache,
            setTaskFileToCache,
            getTaskFileUrlFromApi,
            getTaskFileUrlSync,
            // Podcast 音频缓存
            podcastAudioCache,
            podcastAudioCacheLoaded,
            loadPodcastAudioFromCache,
            savePodcastAudioToCache,
            getPodcastAudioFromCache,
            setPodcastAudioToCache,
            getPodcastAudioUrlFromApi,
            getTemplateFileUrlFromApi,
            getTemplateFileUrl,
            getTemplateFileUrlAsync,
            createTemplateFileUrlRef,
            createTaskFileUrlRef,
            loadTemplateFilesFromCache,
            saveTemplateFilesToCache,
            loadFromCache,
            saveToCache,
            clearAllCache,
            getStatusBadgeClass,
            viewSingleResult,
            cancelTask,
            resumeTask,
            deleteTask,
            startPollingTask,
            stopPollingTask,
            reuseTask,
            showTaskCreator,
            toggleSidebar,
            clearPrompt,
            getTaskItemClass,
            getStatusIndicatorClass,
            getTaskTypeBtnClass,
            getModelBtnClass,
            getTaskTypeIcon,
            getTaskTypeName,
            getPromptPlaceholder,
            getStatusTextClass,
            getImagePreview,
            getTaskInputUrl,
            getTaskInputImage,
            getTaskInputAudio,
            getTaskFileUrl,
            getHistoryImageUrl,
            getUserAvatarUrl,
            getCurrentImagePreviewUrl,
            getCurrentAudioPreviewUrl,
            getCurrentVideoPreviewUrl,
            handleThumbnailError,
            handleImageError,
            handleImageLoad,
            handleAudioError,
            handleAudioLoad,
            getTaskStatusDisplay,
            getTaskStatusColor,
            getTaskStatusIcon,
            getTaskDuration,
            getRelativeTime,
            getTaskHistory,
            getActiveTasks,
            getOverallProgress,
            getProgressTitle,
            getProgressInfo,
            getSubtaskProgress,
            getSubtaskStatusText,
            formatEstimatedTime,
            formatDuration,
            searchTasks,
            filterTasksByStatus,
            filterTasksByType,
            getAlertClass,
            getAlertBorderClass,
            getAlertTextClass,
            getAlertIcon,
            getAlertIconBgClass,
            getPromptTemplates,
            selectPromptTemplate,
            promptHistory,
            getPromptHistory,
            addTaskToHistory,
            getLocalTaskHistory,
            selectPromptHistory,
            clearPromptHistory,
            getImageHistory,
            getAudioHistory,
            selectImageHistory,
            selectAudioHistory,
            previewAudioHistory,
            clearImageHistory,
            clearAudioHistory,
            clearLocalStorage,
            getAudioMimeType,
            getAuthHeaders,
            startResize,
            sidebar,
            switchToCreateView,
            switchToProjectsView,
            switchToInspirationView,
            switchToLoginView,
            openTaskDetailModal,
            closeTaskDetailModal,
            generateShareUrl,
            copyShareLink,
            shareToSocial,
            openTaskFromRoute,
            generateTemplateShareUrl,
            copyTemplateShareLink,
            shareTemplateToSocial,
            openTemplateFromRoute,
            // 灵感广场相关
            inspirationSearchQuery,
            selectedInspirationCategory,
            inspirationItems,
            InspirationCategories,
            loadInspirationData,
            selectInspirationCategory,
            handleInspirationSearch,
            loadMoreInspiration,
            inspirationPagination,
            inspirationPaginationInfo,
            // 精选模版相关
            featuredTemplates,
            featuredTemplatesLoading,
            loadFeaturedTemplates,
            getRandomFeaturedTemplates,
            inspirationCurrentPage,
            inspirationPageSize,
            inspirationPageInput,
            inspirationPaginationKey,
            // 工具函数
            formatDate,
            // 模板详情弹窗相关
            showTemplateDetailModal,
            selectedTemplate,
            previewTemplateDetail,
            closeTemplateDetailModal,
            useTemplate,
            // 图片放大弹窗相关
            showImageZoomModal,
            zoomedImageUrl,
            showImageZoom,
            closeImageZoomModal,
            // 模板素材应用相关
            applyTemplateImage,
            applyTemplateAudio,
            applyTemplatePrompt,
            copyPrompt,
            // 视频播放控制
            playVideo,
            pauseVideo,
            toggleVideoPlay,
            pauseAllVideos,
            updateVideoIcon,
            onVideoLoaded,
            onVideoError,
            onVideoEnded,
            applyMobileStyles,
            handleLoginCallback,
            init,
            validateToken,
            pollingInterval,
            pollingTasks,
            apiRequest,
            // 主题相关
            theme,
            initTheme,
            toggleTheme,
            getThemeIcon,
            loadTtsHistory,
            removeTtsHistoryEntry,
            ttsHistory,
            addTtsHistoryEntry,
            saveTtsHistory,
            clearTtsHistory,
        };
