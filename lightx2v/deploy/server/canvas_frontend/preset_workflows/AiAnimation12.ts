import { WorkflowState, NodeStatus } from '../types';

/** @preset-id preset-ai-animation-12 */
const workflow: WorkflowState = {
  id: 'preset-ai-animation-12',
  name: 'AI动画12分镜日常',
  updatedAt: Date.now(),
  isDirty: false,
  isRunning: false,
  globalInputs: {},
  connections: [
    { id: 'anim-c1', source_node_id: 'anim-char', source_port_id: 'out-image', target_node_id: 'anim-planner', target_port_id: 'in-image' },
    { id: 'anim-c2', source_node_id: 'anim-story', source_port_id: 'out-text', target_node_id: 'anim-planner', target_port_id: 'in-text' },
    { id: 'anim-c3', source_node_id: 'anim-char', source_port_id: 'out-image', target_node_id: 'anim-i2i-1', target_port_id: 'in-image' },
    { id: 'anim-c4', source_node_id: 'anim-planner', source_port_id: 'shot1_image_prompt', target_node_id: 'anim-i2i-1', target_port_id: 'in-text' },
    { id: 'anim-c5', source_node_id: 'anim-i2i-1', source_port_id: 'out-image', target_node_id: 'anim-i2i-2', target_port_id: 'in-image' },
    { id: 'anim-c6', source_node_id: 'anim-planner', source_port_id: 'shot2_image_prompt', target_node_id: 'anim-i2i-2', target_port_id: 'in-text' },
    { id: 'anim-c7', source_node_id: 'anim-i2i-2', source_port_id: 'out-image', target_node_id: 'anim-i2i-3', target_port_id: 'in-image' },
    { id: 'anim-c8', source_node_id: 'anim-planner', source_port_id: 'shot3_image_prompt', target_node_id: 'anim-i2i-3', target_port_id: 'in-text' },
    { id: 'anim-c9', source_node_id: 'anim-i2i-3', source_port_id: 'out-image', target_node_id: 'anim-i2i-4', target_port_id: 'in-image' },
    { id: 'anim-c10', source_node_id: 'anim-planner', source_port_id: 'shot4_image_prompt', target_node_id: 'anim-i2i-4', target_port_id: 'in-text' },
    { id: 'anim-c11', source_node_id: 'anim-i2i-3', source_port_id: 'out-image', target_node_id: 'anim-i2i-5', target_port_id: 'in-image' },
    { id: 'anim-c12', source_node_id: 'anim-planner', source_port_id: 'shot5_image_prompt', target_node_id: 'anim-i2i-5', target_port_id: 'in-text' },
    { id: 'anim-c13', source_node_id: 'anim-i2i-5', source_port_id: 'out-image', target_node_id: 'anim-i2i-6', target_port_id: 'in-image' },
    { id: 'anim-c14', source_node_id: 'anim-planner', source_port_id: 'shot6_image_prompt', target_node_id: 'anim-i2i-6', target_port_id: 'in-text' },
    { id: 'anim-c15', source_node_id: 'anim-i2i-6', source_port_id: 'out-image', target_node_id: 'anim-i2i-7', target_port_id: 'in-image' },
    { id: 'anim-c16', source_node_id: 'anim-planner', source_port_id: 'shot7_image_prompt', target_node_id: 'anim-i2i-7', target_port_id: 'in-text' },
    { id: 'anim-c17', source_node_id: 'anim-char', source_port_id: 'out-image', target_node_id: 'anim-i2i-8', target_port_id: 'in-image' },
    { id: 'anim-c18', source_node_id: 'anim-planner', source_port_id: 'shot8_image_prompt', target_node_id: 'anim-i2i-8', target_port_id: 'in-text' },
    { id: 'anim-c19', source_node_id: 'anim-i2i-8', source_port_id: 'out-image', target_node_id: 'anim-i2i-9', target_port_id: 'in-image' },
    { id: 'anim-c20', source_node_id: 'anim-planner', source_port_id: 'shot9_image_prompt', target_node_id: 'anim-i2i-9', target_port_id: 'in-text' },
    { id: 'anim-c21', source_node_id: 'anim-i2i-9', source_port_id: 'out-image', target_node_id: 'anim-i2i-10', target_port_id: 'in-image' },
    { id: 'anim-c22', source_node_id: 'anim-planner', source_port_id: 'shot10_image_prompt', target_node_id: 'anim-i2i-10', target_port_id: 'in-text' },
    { id: 'anim-c23', source_node_id: 'anim-char', source_port_id: 'out-image', target_node_id: 'anim-i2i-11', target_port_id: 'in-image' },
    { id: 'anim-c24', source_node_id: 'anim-planner', source_port_id: 'shot11_image_prompt', target_node_id: 'anim-i2i-11', target_port_id: 'in-text' },
    { id: 'anim-c25', source_node_id: 'anim-i2i-11', source_port_id: 'out-image', target_node_id: 'anim-i2i-12', target_port_id: 'in-image' },
    { id: 'anim-c26', source_node_id: 'anim-planner', source_port_id: 'shot12_image_prompt', target_node_id: 'anim-i2i-12', target_port_id: 'in-text' },
    { id: 'anim-c27', source_node_id: 'anim-i2i-1', source_port_id: 'out-image', target_node_id: 'anim-v-1', target_port_id: 'in-image' },
    { id: 'anim-c28', source_node_id: 'anim-planner', source_port_id: 'shot1_video_prompt', target_node_id: 'anim-v-1', target_port_id: 'in-text' },
    { id: 'anim-c29', source_node_id: 'anim-i2i-2', source_port_id: 'out-image', target_node_id: 'anim-v-2', target_port_id: 'in-image' },
    { id: 'anim-c30', source_node_id: 'anim-planner', source_port_id: 'shot2_video_prompt', target_node_id: 'anim-v-2', target_port_id: 'in-text' },
    { id: 'anim-c31', source_node_id: 'anim-i2i-3', source_port_id: 'out-image', target_node_id: 'anim-v-3', target_port_id: 'in-image' },
    { id: 'anim-c32', source_node_id: 'anim-planner', source_port_id: 'shot3_video_prompt', target_node_id: 'anim-v-3', target_port_id: 'in-text' },
    { id: 'anim-c33', source_node_id: 'anim-i2i-4', source_port_id: 'out-image', target_node_id: 'anim-v-4', target_port_id: 'in-image-start' },
    { id: 'anim-c34', source_node_id: 'anim-i2i-3', source_port_id: 'out-image', target_node_id: 'anim-v-4', target_port_id: 'in-image-end' },
    { id: 'anim-c35', source_node_id: 'anim-planner', source_port_id: 'shot4_video_prompt', target_node_id: 'anim-v-4', target_port_id: 'in-text' },
    { id: 'anim-c36', source_node_id: 'anim-i2i-4', source_port_id: 'out-image', target_node_id: 'anim-v-5', target_port_id: 'in-image' },
    { id: 'anim-c37', source_node_id: 'anim-planner', source_port_id: 'shot5_video_prompt', target_node_id: 'anim-v-5', target_port_id: 'in-text' },
    { id: 'anim-c38', source_node_id: 'anim-i2i-5', source_port_id: 'out-image', target_node_id: 'anim-v-6', target_port_id: 'in-image' },
    { id: 'anim-c39', source_node_id: 'anim-planner', source_port_id: 'shot6_video_prompt', target_node_id: 'anim-v-6', target_port_id: 'in-text' },
    { id: 'anim-c40', source_node_id: 'anim-i2i-6', source_port_id: 'out-image', target_node_id: 'anim-v-7', target_port_id: 'in-image-start' },
    { id: 'anim-c41', source_node_id: 'anim-i2i-7', source_port_id: 'out-image', target_node_id: 'anim-v-7', target_port_id: 'in-image-end' },
    { id: 'anim-c42', source_node_id: 'anim-planner', source_port_id: 'shot7_video_prompt', target_node_id: 'anim-v-7', target_port_id: 'in-text' },
    { id: 'anim-c43', source_node_id: 'anim-i2i-7', source_port_id: 'out-image', target_node_id: 'anim-v-8', target_port_id: 'in-image' },
    { id: 'anim-c44', source_node_id: 'anim-planner', source_port_id: 'shot8_video_prompt', target_node_id: 'anim-v-8', target_port_id: 'in-text' },
    { id: 'anim-c45', source_node_id: 'anim-i2i-8', source_port_id: 'out-image', target_node_id: 'anim-v-9', target_port_id: 'in-image' },
    { id: 'anim-c46', source_node_id: 'anim-planner', source_port_id: 'shot9_video_prompt', target_node_id: 'anim-v-9', target_port_id: 'in-text' },
    { id: 'anim-c47', source_node_id: 'anim-i2i-9', source_port_id: 'out-image', target_node_id: 'anim-v-10', target_port_id: 'in-image-start' },
    { id: 'anim-c48', source_node_id: 'anim-i2i-10', source_port_id: 'out-image', target_node_id: 'anim-v-10', target_port_id: 'in-image-end' },
    { id: 'anim-c49', source_node_id: 'anim-planner', source_port_id: 'shot10_video_prompt', target_node_id: 'anim-v-10', target_port_id: 'in-text' },
    { id: 'anim-c50', source_node_id: 'anim-i2i-11', source_port_id: 'out-image', target_node_id: 'anim-v-11', target_port_id: 'in-image' },
    { id: 'anim-c51', source_node_id: 'anim-planner', source_port_id: 'shot11_video_prompt', target_node_id: 'anim-v-11', target_port_id: 'in-text' },
    { id: 'anim-c52', source_node_id: 'anim-i2i-12', source_port_id: 'out-image', target_node_id: 'anim-v-12', target_port_id: 'in-image' },
    { id: 'anim-c53', source_node_id: 'anim-planner', source_port_id: 'shot12_video_prompt', target_node_id: 'anim-v-12', target_port_id: 'in-text' }
  ],
  nodes: [
    { id: 'anim-char', tool_id: 'image-input', x: 50, y: 80, status: NodeStatus.IDLE, data: { value: ['/assets/girl.png'] } },
    { id: 'anim-story', tool_id: 'text-input', x: 50, y: 360, status: NodeStatus.IDLE, data: { value: '初音未来的日常，从早上睡醒到晚上睡觉。' } },
    {
      id: 'anim-planner',
      tool_id: 'text-generation',
      x: 400,
      y: 180,
      status: NodeStatus.IDLE,
      data: {
        model: 'doubao-seed-1-6-vision-250815',
        mode: 'custom',
        customInstruction: `你是一位日系二次元动画分镜师。用户会提供一张角色参考图和一段故事概述。请生成12个分镜的详细描述，所有输出字段使用中文。描述要非常细化，明确人物在场景中的具体状态、姿势、表情、动作。

人物一致性（绝对关键）：
- 第一镜：基于用户提供的角色图，描述「该角色在某一场景中的姿态与表情」，保持与参考图一致的发型、发色、服装、五官。
- 第二镜及以后：基于「上一张图」或「用户角色参考图」进行修改（见下方空镜规则），描述「在该新场景/时刻中角色在做什么、表情与动作」，严格保持同一人物的发型、发色、服装特征、五官风格。空镜的下一镜：必须基于「用户提供的角色参考图」描述，不要用空镜图当参考，否则无法延续人物一致性。

场景与风格：
- 场景可自由切换（卧室→窗边→厨房→户外→…），按时间线推进即可。
- 画风统一：全部为日系二次元动画风格；光线与色调随剧情时间自然变化。
- 可有2～3个无人物镜头（空镜）：如窗外云卷云舒、闹钟特写、夕阳空镜等。空镜图必须强调「画面中没有任何人物」「仅场景与氛围」「风格与整体一致」——否则改图工具容易把人物加进去。

输出字段（共24个，只输出该镜头的描述文本，不要加字段名或前缀）：

- shot1_image_prompt ～ shot12_image_prompt（生图）：
- 有角色时：必须明确写出人物的身体姿势与朝向，例如「背影」「侧脸」「正脸」「坐着」「站着」「躺着」等，再描述衣着与在场景中的状态（例如「人物背影，穿着睡衣盖着被子在睡觉」「人物侧脸，背对镜头站卧室窗边，双手正在准备拉开窗帘」）。若是背影或看不到脸的角度，不要写面部表情描述，避免误导模型。
- 空镜时：只描述场景与氛围，明确无人物、仅风格一致。
- 图生视频用的首帧图 = 该分镜下的首帧动作/姿态，即视频第一帧应呈现的状态（如「人物背对镜头站窗边，双手正在准备拉开窗帘」而非该动作结束后的状态）。

- shot1_video_prompt ～ shot12_video_prompt（生视频）：
- 重点描述表情、动作和运镜。例如「睡眼惺忪，睁眼揉了揉眼睛，坐起来打了个大哈欠」。若该镜画面是背影或看不到脸，只描述身体动作与运镜，不要写面部表情，避免误导模型。
- shot4、shot7、shot10 为「场景切换」镜头，本工作流中：shot4 视频 = 空镜（第4张图）作首帧、上一镜有人物图作尾帧，需强调「人物走进画面」以及人物走进画面的具体动作；若某镜是空镜作尾帧、有人物作首帧，则强调「人物走出画面」或「运镜/画面移动到…」完成转场。空镜也可只用单图生视频做运镜（无首尾帧）。
- 人物静止不动的场景（看书、睡觉、发呆、画画等）可用首尾帧表达时间流逝：例如先用图生图从白天改到夜晚，再首帧→尾帧生成表现时间流逝的视频。`,
        custom_outputs: [
          { id: 'shot1_image_prompt', label: '分镜1 画面', description: '第1镜图生图提示' },
          { id: 'shot1_video_prompt', label: '分镜1 运镜', description: '第1镜视频运镜' },
          { id: 'shot2_image_prompt', label: '分镜2 画面', description: '第2镜图生图提示' },
          { id: 'shot2_video_prompt', label: '分镜2 运镜', description: '第2镜视频运镜' },
          { id: 'shot3_image_prompt', label: '分镜3 画面', description: '第3镜图生图提示' },
          { id: 'shot3_video_prompt', label: '分镜3 运镜', description: '第3镜视频运镜' },
          { id: 'shot4_image_prompt', label: '分镜4 画面', description: '第4镜图生图提示（场景切换）' },
          { id: 'shot4_video_prompt', label: '分镜4 运镜', description: '第4镜转场运镜，强调转场' },
          { id: 'shot5_image_prompt', label: '分镜5 画面', description: '第5镜图生图提示' },
          { id: 'shot5_video_prompt', label: '分镜5 运镜', description: '第5镜视频运镜' },
          { id: 'shot6_image_prompt', label: '分镜6 画面', description: '第6镜图生图提示' },
          { id: 'shot6_video_prompt', label: '分镜6 运镜', description: '第6镜视频运镜' },
          { id: 'shot7_image_prompt', label: '分镜7 画面', description: '第7镜图生图提示（场景切换）' },
          { id: 'shot7_video_prompt', label: '分镜7 运镜', description: '第7镜转场运镜，强调转场' },
          { id: 'shot8_image_prompt', label: '分镜8 画面', description: '第8镜图生图提示' },
          { id: 'shot8_video_prompt', label: '分镜8 运镜', description: '第8镜视频运镜' },
          { id: 'shot9_image_prompt', label: '分镜9 画面', description: '第9镜图生图提示' },
          { id: 'shot9_video_prompt', label: '分镜9 运镜', description: '第9镜视频运镜' },
          { id: 'shot10_image_prompt', label: '分镜10 画面', description: '第10镜图生图提示（场景切换）' },
          { id: 'shot10_video_prompt', label: '分镜10 运镜', description: '第10镜转场运镜，强调转场' },
          { id: 'shot11_image_prompt', label: '分镜11 画面', description: '第11镜图生图提示' },
          { id: 'shot11_video_prompt', label: '分镜11 运镜', description: '第11镜视频运镜' },
          { id: 'shot12_image_prompt', label: '分镜12 画面', description: '第12镜图生图提示' },
          { id: 'shot12_video_prompt', label: '分镜12 运镜', description: '第12镜视频运镜' }
        ]
      }
    },
    { id: 'anim-i2i-1', tool_id: 'image-to-image', x: 900, y: 80, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
    { id: 'anim-i2i-2', tool_id: 'image-to-image', x: 900, y: 240, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
    { id: 'anim-i2i-3', tool_id: 'image-to-image', x: 900, y: 400, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
    { id: 'anim-i2i-4', tool_id: 'image-to-image', x: 900, y: 560, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
    { id: 'anim-i2i-5', tool_id: 'image-to-image', x: 900, y: 720, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
    { id: 'anim-i2i-6', tool_id: 'image-to-image', x: 900, y: 880, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
    { id: 'anim-i2i-7', tool_id: 'image-to-image', x: 900, y: 1040, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
    { id: 'anim-i2i-8', tool_id: 'image-to-image', x: 900, y: 1200, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
    { id: 'anim-i2i-9', tool_id: 'image-to-image', x: 900, y: 1360, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
    { id: 'anim-i2i-10', tool_id: 'image-to-image', x: 900, y: 1520, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
    { id: 'anim-i2i-11', tool_id: 'image-to-image', x: 900, y: 1680, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
    { id: 'anim-i2i-12', tool_id: 'image-to-image', x: 900, y: 1840, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
    { id: 'anim-v-1', tool_id: 'video-gen-image', x: 1500, y: 80, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
    { id: 'anim-v-2', tool_id: 'video-gen-image', x: 1500, y: 240, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
    { id: 'anim-v-3', tool_id: 'video-gen-image', x: 1500, y: 400, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
    { id: 'anim-v-4', tool_id: 'video-gen-dual-frame', x: 1500, y: 560, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
    { id: 'anim-v-5', tool_id: 'video-gen-image', x: 1500, y: 720, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
    { id: 'anim-v-6', tool_id: 'video-gen-image', x: 1500, y: 880, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
    { id: 'anim-v-7', tool_id: 'video-gen-dual-frame', x: 1500, y: 1040, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
    { id: 'anim-v-8', tool_id: 'video-gen-image', x: 1500, y: 1200, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
    { id: 'anim-v-9', tool_id: 'video-gen-image', x: 1500, y: 1360, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
    { id: 'anim-v-10', tool_id: 'video-gen-dual-frame', x: 1500, y: 1520, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
    { id: 'anim-v-11', tool_id: 'video-gen-image', x: 1500, y: 1680, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
    { id: 'anim-v-12', tool_id: 'video-gen-image', x: 1500, y: 1840, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } }
  ]
};

export default workflow;
