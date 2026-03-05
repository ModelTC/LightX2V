import { WorkflowState, NodeStatus } from '../types';

/** @preset-id preset-storyboard-9 */
const workflow: WorkflowState = {
  id: 'preset-storyboard-9',
  name: '9分镜故事板视频工作流',
    updatedAt: Date.now(),
    isDirty: false,
    isRunning: false,
    globalInputs: {},
    connections: [
      // Input to first image and planner
      { id: 'c1', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-1', target_port_id: 'in-image' },
      { id: 'c2', source_node_id: 'node-desc', source_port_id: 'out-text', target_node_id: 'node-planner', target_port_id: 'in-text' },
      // Planner to image edits (sequential generation based on previous image + character)
      { id: 'c3', source_node_id: 'node-planner', source_port_id: 'scene1_prompt', target_node_id: 'node-i2i-1', target_port_id: 'in-text' },
      { id: 'c4', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-2', target_port_id: 'in-image' },
      { id: 'c5', source_node_id: 'node-i2i-1', source_port_id: 'out-image', target_node_id: 'node-i2i-2', target_port_id: 'in-image' },
      { id: 'c6', source_node_id: 'node-planner', source_port_id: 'scene2_prompt', target_node_id: 'node-i2i-2', target_port_id: 'in-text' },
      { id: 'c7', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-3', target_port_id: 'in-image' },
      { id: 'c8', source_node_id: 'node-i2i-2', source_port_id: 'out-image', target_node_id: 'node-i2i-3', target_port_id: 'in-image' },
      { id: 'c9', source_node_id: 'node-planner', source_port_id: 'scene3_prompt', target_node_id: 'node-i2i-3', target_port_id: 'in-text' },
      { id: 'c10', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-4', target_port_id: 'in-image' },
      { id: 'c11', source_node_id: 'node-i2i-3', source_port_id: 'out-image', target_node_id: 'node-i2i-4', target_port_id: 'in-image' },
      { id: 'c12', source_node_id: 'node-planner', source_port_id: 'scene4_prompt', target_node_id: 'node-i2i-4', target_port_id: 'in-text' },
      { id: 'c13', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-5', target_port_id: 'in-image' },
      { id: 'c14', source_node_id: 'node-i2i-4', source_port_id: 'out-image', target_node_id: 'node-i2i-5', target_port_id: 'in-image' },
      { id: 'c15', source_node_id: 'node-planner', source_port_id: 'scene5_prompt', target_node_id: 'node-i2i-5', target_port_id: 'in-text' },
      { id: 'c16', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-6', target_port_id: 'in-image' },
      { id: 'c17', source_node_id: 'node-i2i-5', source_port_id: 'out-image', target_node_id: 'node-i2i-6', target_port_id: 'in-image' },
      { id: 'c18', source_node_id: 'node-planner', source_port_id: 'scene6_prompt', target_node_id: 'node-i2i-6', target_port_id: 'in-text' },
      { id: 'c19', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-7', target_port_id: 'in-image' },
      { id: 'c20', source_node_id: 'node-i2i-6', source_port_id: 'out-image', target_node_id: 'node-i2i-7', target_port_id: 'in-image' },
      { id: 'c21', source_node_id: 'node-planner', source_port_id: 'scene7_prompt', target_node_id: 'node-i2i-7', target_port_id: 'in-text' },
      { id: 'c22', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-8', target_port_id: 'in-image' },
      { id: 'c23', source_node_id: 'node-i2i-7', source_port_id: 'out-image', target_node_id: 'node-i2i-8', target_port_id: 'in-image' },
      { id: 'c24', source_node_id: 'node-planner', source_port_id: 'scene8_prompt', target_node_id: 'node-i2i-8', target_port_id: 'in-text' },
      { id: 'c25', source_node_id: 'node-char-img', source_port_id: 'out-image', target_node_id: 'node-i2i-9', target_port_id: 'in-image' },
      { id: 'c26', source_node_id: 'node-i2i-8', source_port_id: 'out-image', target_node_id: 'node-i2i-9', target_port_id: 'in-image' },
      { id: 'c27', source_node_id: 'node-planner', source_port_id: 'scene9_prompt', target_node_id: 'node-i2i-9', target_port_id: 'in-text' },
      // Image to video generation (i2v: each image generates one video)
      { id: 'c28', source_node_id: 'node-i2i-1', source_port_id: 'out-image', target_node_id: 'node-video-1', target_port_id: 'in-image' },
      { id: 'c29', source_node_id: 'node-planner', source_port_id: 'scene1_video', target_node_id: 'node-video-1', target_port_id: 'in-text' },
      { id: 'c30', source_node_id: 'node-i2i-2', source_port_id: 'out-image', target_node_id: 'node-video-2', target_port_id: 'in-image' },
      { id: 'c31', source_node_id: 'node-planner', source_port_id: 'scene2_video', target_node_id: 'node-video-2', target_port_id: 'in-text' },
      { id: 'c32', source_node_id: 'node-i2i-3', source_port_id: 'out-image', target_node_id: 'node-video-3', target_port_id: 'in-image' },
      { id: 'c33', source_node_id: 'node-planner', source_port_id: 'scene3_video', target_node_id: 'node-video-3', target_port_id: 'in-text' },
      { id: 'c34', source_node_id: 'node-i2i-4', source_port_id: 'out-image', target_node_id: 'node-video-4', target_port_id: 'in-image' },
      { id: 'c35', source_node_id: 'node-planner', source_port_id: 'scene4_video', target_node_id: 'node-video-4', target_port_id: 'in-text' },
      { id: 'c36', source_node_id: 'node-i2i-5', source_port_id: 'out-image', target_node_id: 'node-video-5', target_port_id: 'in-image' },
      { id: 'c37', source_node_id: 'node-planner', source_port_id: 'scene5_video', target_node_id: 'node-video-5', target_port_id: 'in-text' },
      { id: 'c38', source_node_id: 'node-i2i-6', source_port_id: 'out-image', target_node_id: 'node-video-6', target_port_id: 'in-image' },
      { id: 'c39', source_node_id: 'node-planner', source_port_id: 'scene6_video', target_node_id: 'node-video-6', target_port_id: 'in-text' },
      { id: 'c40', source_node_id: 'node-i2i-7', source_port_id: 'out-image', target_node_id: 'node-video-7', target_port_id: 'in-image' },
      { id: 'c41', source_node_id: 'node-planner', source_port_id: 'scene7_video', target_node_id: 'node-video-7', target_port_id: 'in-text' },
      { id: 'c42', source_node_id: 'node-i2i-8', source_port_id: 'out-image', target_node_id: 'node-video-8', target_port_id: 'in-image' },
      { id: 'c43', source_node_id: 'node-planner', source_port_id: 'scene8_video', target_node_id: 'node-video-8', target_port_id: 'in-text' },
      { id: 'c44', source_node_id: 'node-i2i-9', source_port_id: 'out-image', target_node_id: 'node-video-9', target_port_id: 'in-image' },
      { id: 'c45', source_node_id: 'node-planner', source_port_id: 'scene9_video', target_node_id: 'node-video-9', target_port_id: 'in-text' }
    ],
    nodes: [
      // Input nodes (vertical spacing 280px to avoid overlap with taller image input)
      { id: 'node-char-img', tool_id: 'image-input', x: 50, y: 360, status: NodeStatus.IDLE, data: { value: ['/assets/princess.png'] } },
      { id: 'node-desc', tool_id: 'text-input', x: 50, y: 80, status: NodeStatus.IDLE, data: { value: "冰雪奇缘中的艾莎公主早晨醒来，在温馨的房间里梳妆打扮，然后望向窗外，窗外是很漂亮的阿伦黛尔小镇风光，然后镜头转向远景能够看到艾莎公主在窗边伸了个懒腰" } },
      // Planner node
      { id: 'node-planner', tool_id: 'text-generation', x: 450, y: 180, status: NodeStatus.IDLE, data: {
          model: 'doubao-seed-1-6-vision-250815',
          mode: 'custom',
          customInstruction: `你是一位专业的视频故事板规划师。你的任务是根据输入描述和人物图片，将其分解为恰好9个场景（分镜）用于顺序图像生成。

关键：人物和场景一致性对这个故事板至关重要。

重要原则：
- 生成结果中的所有字段语言必须跟随用户输入的语言。如果用户使用中文输入，所有输出字段（sceneN_prompt、sceneN_video）都必须使用中文；如果用户使用英文输入，则所有输出字段都使用英文。

图生图修改提示词原则（绝对关键）：
- 强调"改"和"一致性"，不需要描述图片本身的内容
- Scene 1: 将输入的人物图片中的人物改为描述的场景中的样子，保持人物特征的一致性（面部特征、身材、年龄等）
- Scene 2及以后: 基于前一张场景图片，将其中的人物改为新场景中的样子，保持人物特征的一致性。如果人物在场景N改变了服装或发型，后续所有场景都要明确说明"保持与场景N相同的服装/发型"
- 明确说明如何修改：例如"将输入图片（或前一张场景图片）中的人物改为在温馨的房间里梳妆打扮，保持人物特征的一致性"
- 如果场景在同一地点，不要重复描述背景细节，只需说明"保持与前一场景相同的背景"
- 描述需要改变的场景、姿势、表情等
- 强调保持人物特征的一致性（面部特征、身材、年龄、体型等）
- 不要描述光线、背景细节、氛围等图片细节

人物一致性（绝对关键）：
- 所有场景中必须出现相同的人物
- 保持完全相同的人物特征：面部特征（眼睛、鼻子、嘴巴、脸型）、身材、年龄、体型、独特特征
- 发型一致性：如果人物在任何场景中改变了发型（例如从凌乱到整齐，或改变颜色/样式），后续所有场景都必须保持相同的发型/颜色。一旦发型改变，必须在所有后续场景中保持一致。
- 服装一致性：如果人物在任何场景中改变了服装（例如从睡衣到礼服，或从休闲到正式），后续所有场景都必须保持相同的服装。一旦服装改变，必须在所有后续场景中保持一致。在改变后的场景提示词中明确说明"人物穿着与场景N相同的服装"。
- 人物的外观、年龄、体型和视觉风格必须在所有场景中保持完全相同

场景一致性（绝对关键）：
- 如果场景发生在同一地点（例如同一房间、同一户外区域），保持完全相同的背景元素、家具、道具、装饰、光线方向和空间布局
- 如果两个连续场景在同一地点，描述完全相同的背景元素、家具布置和道具，避免视觉不一致
- 保持相同的艺术风格（例如迪士尼风格、写实风格等）在所有场景中
- 保持一致的光线方向、强度和配色方案在整个故事板中
- 确保背景元素和设置在场景之间自然流动
- 保持空间关系和视觉连续性

图像生成方法：
- Scene 1: 基于人物图片生成（建立人物的外观和第一个场景）
- Scene 2: 基于Scene 1图片 + 人物图片生成（保持前一场景的连续性，同时保持人物身份）
- Scene 3: 基于Scene 2图片 + 人物图片生成
- 以此类推... 每个场景使用前一场景图片 + 人物图片以保持连续性和人物一致性

视频生成：
- 每个场景将使用图生视频（i2v）生成
- sceneN_video应该描述该场景的运动/相机运动

对于每个场景，输出：
- sceneN_prompt: 强调"改"和"一致性"的图生图修改提示词，明确说明如何修改输入图片（Scene 1）或前一场景图片（Scene 2+）中的人物，保持人物一致性（特别是服装/发型如果改变 - 明确说明是否穿着与前一场景相同的服装/发型）和场景一致性（如果在同一地点 - 说明保持相同的背景元素）
- sceneN_video: 描述相机运动和动作的视频运动提示词

输出格式：JSON，包含以下字段：
- scene1_prompt, scene1_video
- scene2_prompt, scene2_video
- scene3_prompt, scene3_video
- scene4_prompt, scene4_video
- scene5_prompt, scene5_video
- scene6_prompt, scene6_video
- scene7_prompt, scene7_video
- scene8_prompt, scene8_video
- scene9_prompt, scene9_video

重要：在每个提示词中，明确保持一致性：
- 如果人物在场景N改变了服装，在scene(N+1)_prompt、scene(N+2)_prompt等中说明"人物穿着与场景N相同的[服装描述]"
- 如果人物在场景N改变了发型，在后续提示词中说明"人物保持与场景N相同的发型/发色"
- 如果场景共享同一地点，明确说明"保持与前一场景相同的背景元素、家具、道具和光线"`,
          custom_outputs: [
            { id: 'scene1_prompt', label: 'Scene 1 Image Prompt', description: 'Detailed image prompt for scene 1 with character' },
            { id: 'scene1_video', label: 'Scene 1 Video Prompt', description: 'Video motion prompt for scene 1' },
            { id: 'scene2_prompt', label: 'Scene 2 Image Prompt', description: 'Detailed image prompt for scene 2 with character' },
            { id: 'scene2_video', label: 'Scene 2 Video Prompt', description: 'Video motion prompt for scene 2' },
            { id: 'scene3_prompt', label: 'Scene 3 Image Prompt', description: 'Detailed image prompt for scene 3 with character' },
            { id: 'scene3_video', label: 'Scene 3 Video Prompt', description: 'Video motion prompt for scene 3' },
            { id: 'scene4_prompt', label: 'Scene 4 Image Prompt', description: 'Detailed image prompt for scene 4 with character' },
            { id: 'scene4_video', label: 'Scene 4 Video Prompt', description: 'Video motion prompt for scene 4' },
            { id: 'scene5_prompt', label: 'Scene 5 Image Prompt', description: 'Detailed image prompt for scene 5 with character' },
            { id: 'scene5_video', label: 'Scene 5 Video Prompt', description: 'Video motion prompt for scene 5' },
            { id: 'scene6_prompt', label: 'Scene 6 Image Prompt', description: 'Detailed image prompt for scene 6 with character' },
            { id: 'scene6_video', label: 'Scene 6 Video Prompt', description: 'Video motion prompt for scene 6' },
            { id: 'scene7_prompt', label: 'Scene 7 Prompt', description: 'Image/video prompt for scene 7' },
            { id: 'scene7_video', label: 'Scene 7 Video Prompt', description: 'Video motion prompt for scene 7' },
            { id: 'scene8_prompt', label: 'Scene 8 Prompt', description: 'Image/video prompt for scene 8' },
            { id: 'scene8_video', label: 'Scene 8 Video Prompt', description: 'Video motion prompt for scene 8' },
            { id: 'scene9_prompt', label: 'Scene 9 Prompt', description: 'Image/video prompt for scene 9' },
            { id: 'scene9_video', label: 'Scene 9 Video Prompt', description: 'Video motion prompt for scene 9' }
          ]
      } },
      // Image-to-image nodes for all 9 scenes (vertical spacing 180px)
      { id: 'node-i2i-1', tool_id: 'image-to-image', x: 900, y: 80, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'node-i2i-2', tool_id: 'image-to-image', x: 900, y: 260, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'node-i2i-3', tool_id: 'image-to-image', x: 900, y: 440, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'node-i2i-4', tool_id: 'image-to-image', x: 900, y: 620, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'node-i2i-5', tool_id: 'image-to-image', x: 900, y: 800, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'node-i2i-6', tool_id: 'image-to-image', x: 900, y: 980, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'node-i2i-7', tool_id: 'image-to-image', x: 900, y: 1160, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'node-i2i-8', tool_id: 'image-to-image', x: 900, y: 1340, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'node-i2i-9', tool_id: 'image-to-image', x: 900, y: 1520, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      // Video generation nodes (all 9 scenes, same vertical spacing)
      { id: 'node-video-1', tool_id: 'video-gen-image', x: 1500, y: 80, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
      { id: 'node-video-2', tool_id: 'video-gen-image', x: 1500, y: 260, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
      { id: 'node-video-3', tool_id: 'video-gen-image', x: 1500, y: 440, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
      { id: 'node-video-4', tool_id: 'video-gen-image', x: 1500, y: 620, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
      { id: 'node-video-5', tool_id: 'video-gen-image', x: 1500, y: 800, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
      { id: 'node-video-6', tool_id: 'video-gen-image', x: 1500, y: 980, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
      { id: 'node-video-7', tool_id: 'video-gen-image', x: 1500, y: 1160, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
      { id: 'node-video-8', tool_id: 'video-gen-image', x: 1500, y: 1340, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
      { id: 'node-video-9', tool_id: 'video-gen-image', x: 1500, y: 1520, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } }
    ]
  };

export default workflow;
