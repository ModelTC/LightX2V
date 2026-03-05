import { WorkflowState, NodeStatus } from '../types';

/** @preset-id preset-multi-shot-singing */
const workflow: WorkflowState = {
  id: 'preset-multi-shot-singing',
  name: '数字人多机位唱歌',
    updatedAt: Date.now(),
    isDirty: false,
    isRunning: false,
    globalInputs: {},
    connections: [
      // Input to AI Chat
      { id: 'c1', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-planner', target_port_id: 'in-image' },
      { id: 'c2', source_node_id: 'node-text-in', source_port_id: 'out-text', target_node_id: 'node-planner', target_port_id: 'in-text' },
      // AI Chat to Image-to-Image (9 shots)
      { id: 'c3', source_node_id: 'node-planner', source_port_id: 'shot1_image_prompt', target_node_id: 'node-i2i-1', target_port_id: 'in-text' },
      { id: 'c4', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-1', target_port_id: 'in-image' },
      { id: 'c5', source_node_id: 'node-planner', source_port_id: 'shot2_image_prompt', target_node_id: 'node-i2i-2', target_port_id: 'in-text' },
      { id: 'c6', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-2', target_port_id: 'in-image' },
      { id: 'c7', source_node_id: 'node-planner', source_port_id: 'shot3_image_prompt', target_node_id: 'node-i2i-3', target_port_id: 'in-text' },
      { id: 'c8', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-3', target_port_id: 'in-image' },
      { id: 'c9', source_node_id: 'node-planner', source_port_id: 'shot4_image_prompt', target_node_id: 'node-i2i-4', target_port_id: 'in-text' },
      { id: 'c10', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-4', target_port_id: 'in-image' },
      { id: 'c11', source_node_id: 'node-planner', source_port_id: 'shot5_image_prompt', target_node_id: 'node-i2i-5', target_port_id: 'in-text' },
      { id: 'c12', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-5', target_port_id: 'in-image' },
      { id: 'c28', source_node_id: 'node-planner', source_port_id: 'shot6_image_prompt', target_node_id: 'node-i2i-6', target_port_id: 'in-text' },
      { id: 'c29', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-6', target_port_id: 'in-image' },
      { id: 'c30', source_node_id: 'node-planner', source_port_id: 'shot7_image_prompt', target_node_id: 'node-i2i-7', target_port_id: 'in-text' },
      { id: 'c31', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-7', target_port_id: 'in-image' },
      { id: 'c32', source_node_id: 'node-planner', source_port_id: 'shot8_image_prompt', target_node_id: 'node-i2i-8', target_port_id: 'in-text' },
      { id: 'c33', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-8', target_port_id: 'in-image' },
      { id: 'c34', source_node_id: 'node-planner', source_port_id: 'shot9_image_prompt', target_node_id: 'node-i2i-9', target_port_id: 'in-text' },
      { id: 'c35', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-9', target_port_id: 'in-image' },
      // Image-to-Image to Video Generation (shots 1, 2, 5, 6, 8, 9 -> avatar-gen; shots 3, 4, 7 -> video-gen-image)
      { id: 'c13', source_node_id: 'node-i2i-1', source_port_id: 'out-image', target_node_id: 'node-avatar-1', target_port_id: 'in-image' },
      { id: 'c14', source_node_id: 'node-audio-in', source_port_id: 'out-audio', target_node_id: 'node-avatar-1', target_port_id: 'in-audio' },
      { id: 'c15', source_node_id: 'node-planner', source_port_id: 'shot1_video_prompt', target_node_id: 'node-avatar-1', target_port_id: 'in-text' },
      { id: 'c16', source_node_id: 'node-i2i-2', source_port_id: 'out-image', target_node_id: 'node-avatar-2', target_port_id: 'in-image' },
      { id: 'c17', source_node_id: 'node-audio-in', source_port_id: 'out-audio', target_node_id: 'node-avatar-2', target_port_id: 'in-audio' },
      { id: 'c18', source_node_id: 'node-planner', source_port_id: 'shot2_video_prompt', target_node_id: 'node-avatar-2', target_port_id: 'in-text' },
      // Shot 3 (Wide) - Image-to-Video
      { id: 'c19', source_node_id: 'node-i2i-3', source_port_id: 'out-image', target_node_id: 'node-video-3', target_port_id: 'in-image' },
      { id: 'c21', source_node_id: 'node-planner', source_port_id: 'shot3_video_prompt', target_node_id: 'node-video-3', target_port_id: 'in-text' },
      // Shot 4 (Top-down) - Image-to-Video
      { id: 'c22', source_node_id: 'node-i2i-4', source_port_id: 'out-image', target_node_id: 'node-video-4', target_port_id: 'in-image' },
      { id: 'c24', source_node_id: 'node-planner', source_port_id: 'shot4_video_prompt', target_node_id: 'node-video-4', target_port_id: 'in-text' },
      { id: 'c25', source_node_id: 'node-i2i-5', source_port_id: 'out-image', target_node_id: 'node-avatar-5', target_port_id: 'in-image' },
      { id: 'c26', source_node_id: 'node-audio-in', source_port_id: 'out-audio', target_node_id: 'node-avatar-5', target_port_id: 'in-audio' },
      { id: 'c27', source_node_id: 'node-planner', source_port_id: 'shot5_video_prompt', target_node_id: 'node-avatar-5', target_port_id: 'in-text' },
      { id: 'c36', source_node_id: 'node-i2i-6', source_port_id: 'out-image', target_node_id: 'node-avatar-6', target_port_id: 'in-image' },
      { id: 'c37', source_node_id: 'node-audio-in', source_port_id: 'out-audio', target_node_id: 'node-avatar-6', target_port_id: 'in-audio' },
      { id: 'c38', source_node_id: 'node-planner', source_port_id: 'shot6_video_prompt', target_node_id: 'node-avatar-6', target_port_id: 'in-text' },
      // Shot 7 (Extreme wide) - Image-to-Video
      { id: 'c39', source_node_id: 'node-i2i-7', source_port_id: 'out-image', target_node_id: 'node-video-7', target_port_id: 'in-image' },
      { id: 'c41', source_node_id: 'node-planner', source_port_id: 'shot7_video_prompt', target_node_id: 'node-video-7', target_port_id: 'in-text' },
      // Shot 8 (Over-shoulder) - Image-to-Video
      { id: 'c42', source_node_id: 'node-i2i-8', source_port_id: 'out-image', target_node_id: 'node-video-8', target_port_id: 'in-image' },
      { id: 'c44', source_node_id: 'node-planner', source_port_id: 'shot8_video_prompt', target_node_id: 'node-video-8', target_port_id: 'in-text' },
      { id: 'c45', source_node_id: 'node-i2i-9', source_port_id: 'out-image', target_node_id: 'node-avatar-9', target_port_id: 'in-image' },
      { id: 'c46', source_node_id: 'node-audio-in', source_port_id: 'out-audio', target_node_id: 'node-avatar-9', target_port_id: 'in-audio' },
      { id: 'c47', source_node_id: 'node-planner', source_port_id: 'shot9_video_prompt', target_node_id: 'node-avatar-9', target_port_id: 'in-text' }
    ],
    nodes: [
      // Input nodes (vertical spacing ~200px)
      { id: 'node-char-img', tool_id: 'image-input', x: 50, y: 80, status: NodeStatus.IDLE, data: { value: ['/assets/singing_princess.png'] } },
      { id: 'node-text-in', tool_id: 'text-input', x: 50, y: 360, status: NodeStatus.IDLE, data: { value: "冰雪奇缘中的艾莎公主正在演唱《Let It Go》，动作优雅，表情充满自信和力量" } },
      { id: 'node-audio-in', tool_id: 'audio-input', x: 50, y: 640, status: NodeStatus.IDLE, data: { value: '/assets/let_it_go_part.wav' } },
      // AI Chat Planner (Doubao Vision)
      { id: 'node-planner', tool_id: 'text-generation', x: 450, y: 280, status: NodeStatus.IDLE, data: {
          model: 'doubao-seed-1-6-vision-250815',
          mode: 'custom',
          customInstruction: `你是一位专业的音乐视频多机位导演。你的任务是根据输入的角色图片和可选的文字描述，为演唱表演生成9个不同机位的详细描述。

关键：人物一致性至关重要。相同的人物必须在所有机位中出现，具有完全相同的面部特征、身材、服装和外观。

重要原则：
- 生成结果中的所有字段语言必须跟随用户输入的语言。如果用户使用中文输入，所有输出字段（shotN_image_prompt、shotN_video_prompt）都必须使用中文；如果用户使用英文输入，则所有输出字段都使用英文。

图生图修改提示词原则（绝对关键）：
- 强调"改"和"一致性"，不需要描述图片本身的内容
- 明确说明如何修改输入图片中的人物：例如"将输入图片中的人物改为特写角度，保持人物特征的一致性（面部特征、身材、服装等）"
- 描述需要改变的机位、角度、构图、姿势等
- 强调保持人物特征的一致性（面部特征、身材、服装、外观等）
- 不要描述光线、背景细节、氛围等图片细节

生成9个不同的机位：
1. Shot 1 - 特写: 以脸部为主的镜头，展示详细的面部表情、情感和口型同步
2. Shot 2 - 中景: 上半身镜头，展示头部、肩膀和一些手臂动作。图生图提示词应强调人物的上半身姿势、手势和动态的手臂动作，匹配演唱表演。包括人物上半身服装的细节、肩膀位置以及手臂如何构成画面。
3. Shot 3 - 全景: 全身镜头，展示整个人物和背景。图生图提示词应清楚描述全身姿势、腿部位置、整体肢体语言以及人物如何占据空间。包括完整服装、身体姿势以及人物与背景环境的关系。
4. Shot 4 - 俯拍: 俯视或高角度镜头，从上往下看人物
5. Shot 5 - 侧方位: 侧面或3/4角度镜头，从侧面展示人物
6. Shot 6 - 正面半身: 正面中景镜头，从正面展示人物的上半身，捕捉脸部、肩膀和上躯干。图生图提示词应强调人物的正面外观、面部表情、上半身姿势以及人物如何直接面对镜头。
7. Shot 7 - 极大景: 极宽镜头，在广阔环境中展示人物，强调规模和氛围
8. Shot 8 - 过肩: 过肩镜头，从背后展示人物，营造亲密感
9. Shot 9 - 极特写: 极特写，聚焦眼睛、嘴巴或特定面部特征

对于每个机位，生成：
- shotN_image_prompt: 强调"改"和"一致性"的图生图修改提示词。明确说明如何将输入图片中的人物改为该特定机位和构图，保持完全相同的人物一致性（相同的面部、服装、外观）。描述需要改变的机位角度、构图、姿势/表情。对于机位2、3和6，提供特别详细的身体位置、姿势和空间构图描述。
- shotN_video_prompt: 对于机位1、2、5、6、9（数字人机位）：详细的数字人视频动作描述，包括演唱手势、头部动作、肢体语言、面部表情和与歌曲节奏和能量匹配的动作。对于机位3、4、7、8（图生视频机位）：详细的视频运动描述，包括相机运动技巧（推、拉、摇、移、跟、缩放等）、转场效果和场景动态，以创建机位之间的平滑过渡并与歌曲的节奏和能量匹配。运动描述应专注于相机运动和视觉转场，以在不同机位角度之间创建无缝连接。

输出格式：JSON，包含以下字段：
- shot1_image_prompt, shot1_video_prompt
- shot2_image_prompt, shot2_video_prompt
- shot3_image_prompt, shot3_video_prompt
- shot4_image_prompt, shot4_video_prompt
- shot5_image_prompt, shot5_video_prompt
- shot6_image_prompt, shot6_video_prompt
- shot7_image_prompt, shot7_video_prompt
- shot8_image_prompt, shot8_video_prompt
- shot9_image_prompt, shot9_video_prompt

重要：
- 在所有机位中保持完全相同的人物一致性（相同的面部、服装、外观）
- 每个机位应该有独特的相机角度和构图
- 对于数字人机位（1、2、5、6、9）：视频提示词应描述自然的演唱动作和表情
- 对于图生视频机位（3、4、7、8）：视频提示词必须包括详细的相机运动技巧（推、拉、摇、移、跟、缩放等）和转场效果，以创建机位之间的平滑视觉过渡。这些运动描述对于无缝机位过渡至关重要，应与歌曲的节奏和能量匹配`,
          custom_outputs: [
            { id: 'shot1_image_prompt', label: 'Shot 1 Image Prompt (Close-up)', description: 'Close-up shot image description' },
            { id: 'shot1_video_prompt', label: 'Shot 1 Video Prompt', description: 'Close-up shot video action description' },
            { id: 'shot2_image_prompt', label: 'Shot 2 Image Prompt (Medium)', description: 'Medium shot image description with emphasis on upper body posture, hand gestures, and arm movements' },
            { id: 'shot2_video_prompt', label: 'Shot 2 Video Prompt', description: 'Medium shot video action description' },
            { id: 'shot3_image_prompt', label: 'Shot 3 Image Prompt (Wide)', description: 'Wide shot image description with emphasis on full body pose, leg positioning, and spatial composition' },
            { id: 'shot3_video_prompt', label: 'Shot 3 Video Prompt', description: 'Wide shot video motion description with camera movement and transition effects' },
            { id: 'shot4_image_prompt', label: 'Shot 4 Image Prompt (Top-down)', description: 'Top-down shot image description' },
            { id: 'shot4_video_prompt', label: 'Shot 4 Video Prompt', description: 'Top-down shot video motion description with camera movement and transition effects' },
            { id: 'shot5_image_prompt', label: 'Shot 5 Image Prompt (Side)', description: 'Side angle shot image description' },
            { id: 'shot5_video_prompt', label: 'Shot 5 Video Prompt', description: 'Side angle shot video action description' },
            { id: 'shot6_image_prompt', label: 'Shot 6 Image Prompt (Front medium)', description: 'Front medium shot image description with emphasis on frontal appearance, facial expression, and upper body posture' },
            { id: 'shot6_video_prompt', label: 'Shot 6 Video Prompt', description: 'Low-angle shot video action description' },
            { id: 'shot7_image_prompt', label: 'Shot 7 Image Prompt (Extreme wide)', description: 'Extreme wide shot image description' },
            { id: 'shot7_video_prompt', label: 'Shot 7 Video Prompt', description: 'Extreme wide shot video motion description with camera movement and transition effects' },
            { id: 'shot8_image_prompt', label: 'Shot 8 Image Prompt (Over-shoulder)', description: 'Over-shoulder shot image description' },
            { id: 'shot8_video_prompt', label: 'Shot 8 Video Prompt', description: 'Over-shoulder shot video motion description with camera movement and transition effects' },
            { id: 'shot9_image_prompt', label: 'Shot 9 Image Prompt (Extreme close-up)', description: 'Extreme close-up shot image description' },
            { id: 'shot9_video_prompt', label: 'Shot 9 Video Prompt', description: 'Extreme close-up shot video action description' }
          ]
      } },
      // Image-to-Image nodes (9 shots, vertical spacing 180px)
      { id: 'node-i2i-1', tool_id: 'image-to-image', x: 900, y: 80, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'node-i2i-2', tool_id: 'image-to-image', x: 900, y: 260, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'node-i2i-3', tool_id: 'image-to-image', x: 900, y: 440, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'node-i2i-4', tool_id: 'image-to-image', x: 900, y: 620, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'node-i2i-5', tool_id: 'image-to-image', x: 900, y: 800, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'node-i2i-6', tool_id: 'image-to-image', x: 900, y: 980, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'node-i2i-7', tool_id: 'image-to-image', x: 900, y: 1160, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'node-i2i-8', tool_id: 'image-to-image', x: 900, y: 1340, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'node-i2i-9', tool_id: 'image-to-image', x: 900, y: 1520, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      // Avatar Video nodes (shots 1, 2, 5, 6, 9 - digital avatar, same y as i2i)
      { id: 'node-avatar-1', tool_id: 'avatar-gen', x: 1500, y: 80, status: NodeStatus.IDLE, data: {} },
      { id: 'node-avatar-2', tool_id: 'avatar-gen', x: 1500, y: 260, status: NodeStatus.IDLE, data: {} },
      { id: 'node-avatar-5', tool_id: 'avatar-gen', x: 1500, y: 800, status: NodeStatus.IDLE, data: {} },
      { id: 'node-avatar-6', tool_id: 'avatar-gen', x: 1500, y: 980, status: NodeStatus.IDLE, data: {} },
      { id: 'node-avatar-9', tool_id: 'avatar-gen', x: 1500, y: 1520, status: NodeStatus.IDLE, data: {} },
      // Image-to-Video nodes (shots 3, 4, 7, 8 - image-to-video, same y as i2i)
      { id: 'node-video-3', tool_id: 'video-gen-image', x: 1500, y: 440, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
      { id: 'node-video-4', tool_id: 'video-gen-image', x: 1500, y: 620, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
      { id: 'node-video-7', tool_id: 'video-gen-image', x: 1500, y: 1160, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
      { id: 'node-video-8', tool_id: 'video-gen-image', x: 1500, y: 1340, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } }
    ]
  };

export default workflow;
