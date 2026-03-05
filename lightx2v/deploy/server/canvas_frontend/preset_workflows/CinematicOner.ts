import { WorkflowState, NodeStatus } from '../types';

/** @preset-id preset-cinematic-oner */
const workflow: WorkflowState = {
  id: 'preset-cinematic-oner',
  name: '大师级运镜一镜到底首尾帧视频工作流',
    updatedAt: Date.now(),
    isDirty: false,
    isRunning: false,
    globalInputs: {},
    connections: [
      // Input to planner
      { id: 'oner-c1', source_node_id: 'oner-node-desc', source_port_id: 'out-text', target_node_id: 'oner-node-planner', target_port_id: 'in-text' },
      // Planner to image generation (5 cinematic shots)
      { id: 'oner-c2', source_node_id: 'oner-node-planner', source_port_id: 'shot1_image_prompt', target_node_id: 'oner-node-img-1', target_port_id: 'in-text' },
      { id: 'oner-c3', source_node_id: 'oner-node-img-1', source_port_id: 'out-image', target_node_id: 'oner-node-img-2', target_port_id: 'in-image' },
      { id: 'oner-c4', source_node_id: 'oner-node-planner', source_port_id: 'shot2_image_prompt', target_node_id: 'oner-node-img-2', target_port_id: 'in-text' },
      { id: 'oner-c5', source_node_id: 'oner-node-img-2', source_port_id: 'out-image', target_node_id: 'oner-node-img-3', target_port_id: 'in-image' },
      { id: 'oner-c6', source_node_id: 'oner-node-planner', source_port_id: 'shot3_image_prompt', target_node_id: 'oner-node-img-3', target_port_id: 'in-text' },
      { id: 'oner-c7', source_node_id: 'oner-node-img-3', source_port_id: 'out-image', target_node_id: 'oner-node-img-4', target_port_id: 'in-image' },
      { id: 'oner-c8', source_node_id: 'oner-node-planner', source_port_id: 'shot4_image_prompt', target_node_id: 'oner-node-img-4', target_port_id: 'in-text' },
      { id: 'oner-c9', source_node_id: 'oner-node-img-4', source_port_id: 'out-image', target_node_id: 'oner-node-img-5', target_port_id: 'in-image' },
      { id: 'oner-c10', source_node_id: 'oner-node-planner', source_port_id: 'shot5_image_prompt', target_node_id: 'oner-node-img-5', target_port_id: 'in-text' },
      // Image to video generation (dual-frame: start frame + end frame)
      // Video 1: shot1 (start) -> shot2 (end)
      { id: 'oner-c11', source_node_id: 'oner-node-img-1', source_port_id: 'out-image', target_node_id: 'oner-node-video-1', target_port_id: 'in-image-start' },
      { id: 'oner-c11b', source_node_id: 'oner-node-img-2', source_port_id: 'out-image', target_node_id: 'oner-node-video-1', target_port_id: 'in-image-end' },
      { id: 'oner-c12', source_node_id: 'oner-node-planner', source_port_id: 'shot1_video_motion', target_node_id: 'oner-node-video-1', target_port_id: 'in-text' },
      // Video 2: shot2 (start) -> shot3 (end)
      { id: 'oner-c13', source_node_id: 'oner-node-img-2', source_port_id: 'out-image', target_node_id: 'oner-node-video-2', target_port_id: 'in-image-start' },
      { id: 'oner-c13b', source_node_id: 'oner-node-img-3', source_port_id: 'out-image', target_node_id: 'oner-node-video-2', target_port_id: 'in-image-end' },
      { id: 'oner-c14', source_node_id: 'oner-node-planner', source_port_id: 'shot2_video_motion', target_node_id: 'oner-node-video-2', target_port_id: 'in-text' },
      // Video 3: shot3 (start) -> shot4 (end)
      { id: 'oner-c15', source_node_id: 'oner-node-img-3', source_port_id: 'out-image', target_node_id: 'oner-node-video-3', target_port_id: 'in-image-start' },
      { id: 'oner-c15b', source_node_id: 'oner-node-img-4', source_port_id: 'out-image', target_node_id: 'oner-node-video-3', target_port_id: 'in-image-end' },
      { id: 'oner-c16', source_node_id: 'oner-node-planner', source_port_id: 'shot3_video_motion', target_node_id: 'oner-node-video-3', target_port_id: 'in-text' },
      // Video 4: shot4 (start) -> shot5 (end)
      { id: 'oner-c17', source_node_id: 'oner-node-img-4', source_port_id: 'out-image', target_node_id: 'oner-node-video-4', target_port_id: 'in-image-start' },
      { id: 'oner-c17b', source_node_id: 'oner-node-img-5', source_port_id: 'out-image', target_node_id: 'oner-node-video-4', target_port_id: 'in-image-end' },
      { id: 'oner-c18', source_node_id: 'oner-node-planner', source_port_id: 'shot4_video_motion', target_node_id: 'oner-node-video-4', target_port_id: 'in-text' },
      // Video 5: shot5 (start) -> shot5 (end, same frame for final shot)
      { id: 'oner-c19', source_node_id: 'oner-node-img-5', source_port_id: 'out-image', target_node_id: 'oner-node-video-5', target_port_id: 'in-image-start' },
      { id: 'oner-c19b', source_node_id: 'oner-node-img-5', source_port_id: 'out-image', target_node_id: 'oner-node-video-5', target_port_id: 'in-image-end' },
      { id: 'oner-c20', source_node_id: 'oner-node-planner', source_port_id: 'shot5_video_motion', target_node_id: 'oner-node-video-5', target_port_id: 'in-text' }
    ],
    nodes: [
      { id: 'oner-node-desc', tool_id: 'text-input', x: 50, y: 280, status: NodeStatus.IDLE, data: { value: "一座未来主义赛博朋克城市的宏大全景，从高空俯瞰整座城市，镜头逐渐下降穿过云雾，掠过摩天大楼的玻璃幕墙，最终聚焦到繁华街道上的人群和霓虹灯" } },
      { id: 'oner-node-planner', tool_id: 'text-generation', x: 450, y: 280, status: NodeStatus.IDLE, data: {
          model: 'deepseek-v3-2-251201',
          mode: 'custom',
          customInstruction: `你是一位专业的电影级视频分镜设计师。你的任务是根据用户的场景描述，设计一个"一镜到底"的连续运镜视频。

重要原则：
- 生成结果中的所有字段语言必须跟随用户输入的语言。如果用户使用中文输入，所有输出字段（shotN_image_prompt、shotN_video_motion）都必须使用中文；如果用户使用英文输入，则所有输出字段都使用英文。
- 这是"一镜到底"视频，所有镜头必须形成连续、流畅的视觉过渡
- 运用多种电影级运镜技巧：极远景、全景、中景、近景、特写、推拉镜头、跟随镜头、环绕镜头、升降镜头、俯仰镜头等
- 场景必须宏大、震撼，具有视觉冲击力
- 保持场景一致性和视觉连贯性（同一场景的不同视角）
- 每个分镜的视觉风格、色调、氛围必须统一
- 镜头运动要流畅自然，前后分镜要有逻辑连接

图生图修改提示词原则（对于分镜2-5）：
- 强调"改"和"一致性"，不需要描述图片本身的内容
- Shot 1: 使用文生图，详细描述画面内容、构图、光线、氛围
- Shot 2-5: 明确说明如何修改前一张分镜图片：例如"将前一张分镜图片改为中景构图，镜头逐渐接近，保持场景元素的一致性"
- 描述需要改变的构图、视角、镜头位置等
- 强调保持场景元素的一致性（如果是同一个场景，要说明"保持与前一分镜相同的场景元素"）
- 不要重复描述光线、背景细节、氛围等图片细节

输出要求：
- 生成5个分镜，每个分镜包含：
  - shotN_image_prompt: Shot 1使用文生图提示词，详细描述画面内容、构图（构图规则，如三分法、对称等）、光线（自然光、人工光、色调、明暗对比）、氛围和情绪。Shot 2-5使用强调"改"和"一致性"的图生图修改提示词，明确说明如何修改前一张分镜图片，保持场景元素的连贯性。
  - shotN_video_motion: 详细的视频运动提示词，描述该分镜的相机运动、运镜方式、运动方向、速度

分镜设计思路（参考）：
- 分镜1: 极远景/全景 - 建立宏大场景，可能是高空俯瞰或全景展示（文生图）
- 分镜2: 中景/推拉 - 镜头逐渐接近，或从一侧移动到另一侧（图生图：基于分镜1）
- 分镜3: 中景/跟随/环绕 - 镜头运动，展现场景的深度和细节（图生图：基于分镜2）
- 分镜4: 近景/特写 - 聚焦到场景中的关键元素或细节（图生图：基于分镜3）
- 分镜5: 全景/极远景收尾 - 回到宏大的视角，形成视觉闭环（图生图：基于分镜4）

每个video_motion应该（关键：这是首尾帧生视频，强调从当前镜头到下一个镜头的运镜切换）：
- 使用电影式运镜语言，描述从当前分镜（首帧）到下一个分镜（尾帧）的相机运动
- 重点描述运镜切换：如何从当前镜头的构图、角度、位置，通过相机运动过渡到下一个镜头的构图、角度、位置
- 运用电影级运镜术语：
  * 推拉镜头（Dolly In/Out）：相机前后移动
  * 摇镜头（Pan）：相机左右旋转
  * 移镜头（Truck）：相机左右平移
  * 升降镜头（Crane/Boom）：相机上下移动
  * 跟随镜头（Follow/Tracking）：相机跟随主体移动
  * 环绕镜头（Orbit/Arc）：相机围绕主体旋转
  * 变焦（Zoom In/Out）：镜头焦距变化
  * 俯仰镜头（Tilt）：相机上下旋转
- 描述运动速度（慢速、中速、快速）和节奏感
- 描述运动轨迹和路径（直线、弧线、曲线等）
- 强调视觉连贯性：确保从首帧到尾帧的过渡自然流畅，形成"一镜到底"的连续感
- 对于最后一个分镜（shot5），可以描述一个收尾性的运镜，如缓慢拉远或环绕收尾

输出JSON格式，包含以下字段：
- shot1_image_prompt, shot1_video_motion
- shot2_image_prompt, shot2_video_motion
- shot3_image_prompt, shot3_video_motion
- shot4_image_prompt, shot4_video_motion
- shot5_image_prompt, shot5_video_motion`,
          custom_outputs: [
            { id: 'shot1_image_prompt', label: '分镜1图像提示', description: '第一分镜的详细图像生成提示词（远景/全景，建立宏大场景）' },
            { id: 'shot1_video_motion', label: '分镜1运镜描述', description: '从分镜1到分镜2的运镜切换提示词，使用电影式运镜语言描述相机运动' },
            { id: 'shot2_image_prompt', label: '分镜2图像提示', description: '第二分镜的详细图像生成提示词（中景，镜头逐渐接近）' },
            { id: 'shot2_video_motion', label: '分镜2运镜描述', description: '从分镜2到分镜3的运镜切换提示词，使用电影式运镜语言描述相机运动' },
            { id: 'shot3_image_prompt', label: '分镜3图像提示', description: '第三分镜的详细图像生成提示词（中景/跟随，展现深度）' },
            { id: 'shot3_video_motion', label: '分镜3运镜描述', description: '从分镜3到分镜4的运镜切换提示词，使用电影式运镜语言描述相机运动' },
            { id: 'shot4_image_prompt', label: '分镜4图像提示', description: '第四分镜的详细图像生成提示词（近景/特写，聚焦细节）' },
            { id: 'shot4_video_motion', label: '分镜4运镜描述', description: '从分镜4到分镜5的运镜切换提示词，使用电影式运镜语言描述相机运动' },
            { id: 'shot5_image_prompt', label: '分镜5图像提示', description: '第五分镜的详细图像生成提示词（全景收尾，视觉闭环）' },
            { id: 'shot5_video_motion', label: '分镜5运镜描述', description: '分镜5的收尾运镜提示词，使用电影式运镜语言描述收尾性的相机运动' }
          ]
      } },
      { id: 'oner-node-img-1', tool_id: 'text-to-image', x: 900, y: 80, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-2512', aspectRatio: "16:9" } },
      { id: 'oner-node-img-2', tool_id: 'image-to-image', x: 900, y: 260, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: "16:9" } },
      { id: 'oner-node-img-3', tool_id: 'image-to-image', x: 900, y: 440, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: "16:9" } },
      { id: 'oner-node-img-4', tool_id: 'image-to-image', x: 900, y: 620, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: "16:9" } },
      { id: 'oner-node-img-5', tool_id: 'image-to-image', x: 900, y: 800, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: "16:9" } },
      { id: 'oner-node-video-1', tool_id: 'video-gen-dual-frame', x: 1500, y: 80, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: "16:9" } },
      { id: 'oner-node-video-2', tool_id: 'video-gen-dual-frame', x: 1500, y: 260, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: "16:9" } },
      { id: 'oner-node-video-3', tool_id: 'video-gen-dual-frame', x: 1500, y: 440, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: "16:9" } },
      { id: 'oner-node-video-4', tool_id: 'video-gen-dual-frame', x: 1500, y: 620, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: "16:9" } },
      { id: 'oner-node-video-5', tool_id: 'video-gen-dual-frame', x: 1500, y: 800, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: "16:9" } }
    ]
  };

export default workflow;
