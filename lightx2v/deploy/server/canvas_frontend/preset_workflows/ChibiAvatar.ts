import { WorkflowState, NodeStatus } from '../types';

/** @preset-id preset-chibi-avatar */
const workflow: WorkflowState = {
  id: 'preset-chibi-avatar',
  name: 'Q版数字人',
    updatedAt: Date.now(),
    isDirty: false,
    isRunning: false,
    globalInputs: {},
    connections: [
      { id: 'c1', source_node_id: 'chibi-node-person', source_port_id: 'out-image', target_node_id: 'chibi-node-ai', target_port_id: 'in-image' },
      { id: 'c2', source_node_id: 'chibi-node-input', source_port_id: 'out-text', target_node_id: 'chibi-node-ai', target_port_id: 'in-text' },
      { id: 'c3', source_node_id: 'chibi-node-ai', source_port_id: 'i2i_prompt', target_node_id: 'chibi-node-i2i', target_port_id: 'in-text' },
      { id: 'c4', source_node_id: 'chibi-node-person', source_port_id: 'out-image', target_node_id: 'chibi-node-i2i', target_port_id: 'in-image' },
      { id: 'c5', source_node_id: 'chibi-node-ai', source_port_id: 'tts_text', target_node_id: 'chibi-node-tts', target_port_id: 'in-text' },
      { id: 'c6', source_node_id: 'chibi-node-ai', source_port_id: 'voice_style', target_node_id: 'chibi-node-tts', target_port_id: 'in-context-tone' },
      { id: 'c7', source_node_id: 'chibi-node-i2i', source_port_id: 'out-image', target_node_id: 'chibi-node-avatar', target_port_id: 'in-image' },
      { id: 'c8', source_node_id: 'chibi-node-tts', source_port_id: 'out-audio', target_node_id: 'chibi-node-avatar', target_port_id: 'in-audio' },
      { id: 'c9', source_node_id: 'chibi-node-ai', source_port_id: 's2v_prompt', target_node_id: 'chibi-node-avatar', target_port_id: 'in-text' }
    ],
    nodes: [
      { id: 'chibi-node-doll-ref', tool_id: 'image-input', x: 50, y: 80, status: NodeStatus.IDLE, data: { value: ['/assets/doll.jpg'] } },
      { id: 'chibi-node-person', tool_id: 'image-input', x: 50, y: 360, status: NodeStatus.IDLE, data: { value: [] } },
      { id: 'chibi-node-input', tool_id: 'text-input', x: 50, y: 640, status: NodeStatus.IDLE, data: { value: "角色说的话和角色背景描述" } },
      { id: 'chibi-node-ai', tool_id: 'text-generation', x: 450, y: 280, status: NodeStatus.IDLE, data: {
          model: 'doubao-seed-1-6-vision-250815',
          mode: 'custom',
          customInstruction: `你是一位专业的Q版数字人视频创意总监。你的任务是根据输入的人物图片和用户描述（角色说的话和角色背景），生成完整的Q版数字人视频组件。

重要原则：
- 生成结果中的所有字段语言必须跟随用户输入的语言。如果用户使用中文输入，所有输出字段（i2i_prompt、tts_text、voice_style、s2v_prompt）都必须使用中文；如果用户使用英文输入，则所有输出字段都使用英文。

参考风格：Q版球关节娃娃风格，将照片中的人物转换为Q版球关节娃娃，保持人物的面部特征一致但风格化，大脑袋小身体比例，光滑的树脂质感皮肤，闪亮的娃娃眼睛，简化的面部结构，柔和的腮红，可见的关节段，可爱的艺术玩具美学，角色自然站立，双脚着地，干净的纯白背景，柔和均匀的光线，高细节，无阴影杂乱。

对于 i2i_prompt（图生图修改提示词）：
- 基于输入的人物图片，将其转换为Q版球关节娃娃风格
- 强调"改"和"一致性"，不需要描述图片本身的内容
- 明确说明如何修改：将输入图片中的人物转换为Q版球关节娃娃风格，保持人物面部特征的一致性（面部特征、表情等），但风格化为大脑袋小身体比例，光滑的树脂质感皮肤，闪亮的娃娃眼睛，简化的面部结构，柔和的腮红，可见的关节段，可爱的艺术玩具美学
- 角色自然站立，双脚着地，干净的纯白背景，柔和均匀的光线，高细节，无阴影杂乱
- 确保修改后的描述与角色背景相匹配

对于 tts_text（语音脚本）：
- 根据用户的输入描述生成相应的口语脚本
- 脚本内容应与用户的描述和角色背景相匹配
- 使用自然、对话式的语言
- 长度应适合数字人视频使用（正常语速下约20-40秒）

对于 voice_style（语调指令）：
- 根据用户的输入描述和角色背景特点，提供相应的配音指导
- 描述配音应该体现的风格、情感、语气等特点
- 确保语调指令与脚本内容和角色背景相匹配
- Q版角色通常需要可爱、活泼、轻快的语调

对于 s2v_prompt（数字人视频动作提示）：
- 描述自然、真实的说话手势和动作
- 包含头部动作（点头、倾斜、轻微转动）
- 描述与语音内容和情感语调匹配的面部表情
- Q版角色可以有一些可爱的动作，如轻微摆动、活泼的手势
- 指定眼神接触和视线方向
- 确保动作与语音节奏同步
- 足够详细，以指导数字人视频生成自然、逼真的效果`,
          custom_outputs: [
            { id: 'i2i_prompt', label: '改图提示词', description: '将人物转换为Q版球关节娃娃风格的图生图修改提示词。' },
            { id: 'tts_text', label: 'TTS文案', description: '根据用户描述生成的语音脚本。' },
            { id: 'voice_style', label: 'TTS语气指令', description: '根据用户描述和角色背景生成的配音指导。' },
            { id: 's2v_prompt', label: 'S2V提示词', description: '数字人视频动作和运动的描述。' }
          ]
      } },
      { id: 'chibi-node-i2i', tool_id: 'image-to-image', x: 900, y: 80, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'chibi-node-tts', tool_id: 'tts', x: 900, y: 440, status: NodeStatus.IDLE, data: { model: 'lightx2v', voiceType: 'zh_female_vv_uranus_bigtts', resourceId: 'seed-tts-2.0' } },
      { id: 'chibi-node-avatar', tool_id: 'avatar-gen', x: 1500, y: 260, status: NodeStatus.IDLE, data: { model: 'SekoTalk' } }
    ]
  };

export default workflow;
