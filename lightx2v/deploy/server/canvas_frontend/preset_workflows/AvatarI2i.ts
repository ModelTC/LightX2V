import { WorkflowState, NodeStatus } from '../types';

/** @preset-id preset-avatar-i2i */
const workflow: WorkflowState = {
  id: 'preset-avatar-i2i',
  name: '无限数字人口播视频工作流',
    updatedAt: Date.now(),
    isDirty: false,
    isRunning: false,
    globalInputs: {},
    connections: [
      { id: 'c1', source_node_id: 'node-img-in', source_port_id: 'out-image', target_node_id: 'node-i2i-gen', target_port_id: 'in-image' },
      { id: 'c2', source_node_id: 'node-text-in', source_port_id: 'out-text', target_node_id: 'node-logic', target_port_id: 'in-text' },
      { id: 'c3', source_node_id: 'node-logic', source_port_id: 'i2i_prompt', target_node_id: 'node-i2i-gen', target_port_id: 'in-text' },
      { id: 'c4', source_node_id: 'node-logic', source_port_id: 'tts_text', target_node_id: 'node-voice', target_port_id: 'in-text' },
      { id: 'c5', source_node_id: 'node-logic', source_port_id: 'voice_style', target_node_id: 'node-voice', target_port_id: 'in-context-tone' },
      { id: 'c6', source_node_id: 'node-i2i-gen', source_port_id: 'out-image', target_node_id: 'node-final-avatar', target_port_id: 'in-image' },
      { id: 'c7', source_node_id: 'node-voice', source_port_id: 'out-audio', target_node_id: 'node-final-avatar', target_port_id: 'in-audio' }
    ],
    nodes: [
      { id: 'node-img-in', tool_id: 'image-input', x: 50, y: 80, status: NodeStatus.IDLE, data: { value: ['/assets/girl.jpg'] } },
      { id: 'node-text-in', tool_id: 'text-input', x: 50, y: 360, status: NodeStatus.IDLE, data: { value: "女孩改为穿着性感纯欲的睡衣坐在床上，用性感迷人的声音说着勾人的话" } },
      { id: 'node-logic', tool_id: 'text-generation', x: 450, y: 280, status: NodeStatus.IDLE, data: {
          model: 'doubao-seed-1-6-vision-250815',
          mode: 'custom',
          customInstruction: `你是一位专业的数字人视频创意总监。你的任务是根据输入的人物图片和文字描述，生成同步的数字人视频组件。

重要原则：
- 生成结果中的所有字段语言必须跟随用户输入的语言。如果用户使用中文输入，所有输出字段（i2i_prompt、tts_text、voice_style）都必须使用中文；如果用户使用英文输入，则所有输出字段都使用英文。

对于 i2i_prompt（图生图修改提示词）：
- 强调"改"和"一致性"，不需要描述图片本身的内容
- 明确说明如何修改输入图片中的人物：例如"将输入图片中的人物改为穿着性感纯欲的睡衣坐在床上，保持人物特征的一致性（面部特征、身材等）"
- 描述需要改变的服装、场景、姿势、表情等
- 强调保持人物特征的一致性（面部特征、身材、年龄等）
- 确保修改后的描述与脚本内容相匹配

对于 tts_text（语音脚本）：
- 根据用户的输入描述生成相应的口语脚本
- 脚本内容应与用户的描述和场景相匹配
- 使用自然、对话式的语言
- 长度应适合数字人视频使用

对于 voice_style（语调指令）：
- 根据用户的输入描述和场景特点，提供相应的配音指导
- 描述配音应该体现的风格、情感、语气等特点
- 确保语调指令与脚本内容相匹配`,
          custom_outputs: [
            { id: 'i2i_prompt', label: '场景修改提示词', description: '图生图修改提示词。强调"改"和"一致性"，明确说明如何修改输入图片中的人物。' },
            { id: 'tts_text', label: '数字人脚本', description: '根据用户描述生成的口语脚本，与场景相匹配。' },
            { id: 'voice_style', label: '语调', description: '根据用户描述和场景特点生成的配音指导，描述应该体现的风格、情感、语气等。' }
          ]
      } },
      { id: 'node-i2i-gen', tool_id: 'image-to-image', x: 900, y: 80, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'node-voice', tool_id: 'tts', x: 900, y: 440, status: NodeStatus.IDLE, data: { model: 'lightx2v', voiceType: 'zh_female_vv_uranus_bigtts', resourceId: 'seed-tts-2.0' } },
      { id: 'node-final-avatar', tool_id: 'avatar-gen', x: 1500, y: 260, status: NodeStatus.IDLE, data: {} }
    ]
  };

export default workflow;
