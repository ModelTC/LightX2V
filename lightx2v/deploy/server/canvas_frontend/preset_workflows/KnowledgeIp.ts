import { WorkflowState, NodeStatus } from '../types';

/** @preset-id preset-knowledge-ip */
const workflow: WorkflowState = {
  id: 'preset-knowledge-ip',
  name: '知识IP口播工作流',
  updatedAt: Date.now(),
  isDirty: false,
  isRunning: false,
  globalInputs: {},
  connections: [
    { id: 'c1', source_node_id: 'ip-node-url', source_port_id: 'out-text', target_node_id: 'ip-node-ai', target_port_id: 'in-text' },
    { id: 'c2', source_node_id: 'ip-node-image-ref', source_port_id: 'out-image', target_node_id: 'ip-node-ai', target_port_id: 'in-image' },
    { id: 'c3', source_node_id: 'ip-node-ai', source_port_id: 'tts_text', target_node_id: 'ip-node-tts', target_port_id: 'in-text' },
    { id: 'c4', source_node_id: 'ip-node-ai', source_port_id: 'voice_style', target_node_id: 'ip-node-tts', target_port_id: 'in-context-tone' },
    { id: 'c5', source_node_id: 'ip-node-image-ref', source_port_id: 'out-image', target_node_id: 'ip-node-avatar', target_port_id: 'in-image' },
    { id: 'c6', source_node_id: 'ip-node-tts', source_port_id: 'out-audio', target_node_id: 'ip-node-avatar', target_port_id: 'in-audio' },
    { id: 'c7', source_node_id: 'ip-node-ai', source_port_id: 's2v_prompt', target_node_id: 'ip-node-avatar', target_port_id: 'in-text' }
  ],
  nodes: [
    { id: 'ip-node-url', tool_id: 'text-input', x: 80, y: 80, status: NodeStatus.IDLE, data: { value: "https://github.com/ModelTC/LightX2V/blob/main/README_zh.md" } },
    { id: 'ip-node-image-ref', tool_id: 'image-input', x: 80, y: 360, status: NodeStatus.IDLE, data: { value: ['/assets/programmer.png'] } },
    { id: 'ip-node-ai', tool_id: 'text-generation', x: 520, y: 180, status: NodeStatus.IDLE, data: {
        model: 'doubao-seed-1-6-vision-250815',
        mode: 'custom',
        useSearch: true,
        customInstruction: `你是一位专业的知识IP口播视频创意总监。你的任务是根据输入的网页链接内容和数字人参考图片，生成完整的知识IP口播视频组件。

重要原则：
- 生成结果中的所有字段语言必须跟随用户输入的语言。如果用户使用中文输入，所有输出字段（tts_text、voice_style、s2v_prompt）都必须使用中文；如果用户使用英文输入，则所有输出字段都使用英文。
- 你需要访问用户提供的网页链接，提取网页的核心内容，理解其主题和关键信息。使用联网搜索功能获取网页内容。
- 用户会提供参考图片，直接使用该图片作为数字人形象，不需要生成或修改图片。

对于 tts_text（口播文案）：
- 根据网页链接的内容，提取核心知识点和关键信息
- 将网页内容转化为自然、流畅的口播文案
- 文案应该结构清晰，逻辑性强，易于理解
- 使用口语化的表达方式，适合口播
- 长度应适合数字人视频使用（正常语速下约30-60秒）
- 确保文案准确传达网页的核心内容，同时保持吸引力和可理解性

对于 voice_style（语气指令）：
- 根据知识IP的定位和内容特点，提供相应的配音指导
- 知识IP通常需要专业、清晰、有说服力的语调
- 描述配音应该体现的风格、情感、语气等特点
- 确保语调指令与口播文案内容相匹配
- 可以包含节奏、重音、停顿等具体指导

对于 s2v_prompt（数字人视频动作提示）：
- 描述自然、真实的说话手势和动作
- 包含头部动作（点头、倾斜、轻微转动）
- 描述与语音内容和情感语调匹配的面部表情
- 知识IP可以有一些专业的手势，如指向、展示等
- 指定眼神接触和视线方向
- 确保动作与语音节奏同步
- 足够详细，以指导数字人视频生成自然、逼真的效果`,
        custom_outputs: [
          { id: 'tts_text', label: '口播文案', description: '根据网页内容生成的口播文案。' },
          { id: 'voice_style', label: '语气指令', description: '根据知识IP定位生成的配音指导。' },
          { id: 's2v_prompt', label: 'S2V提示词', description: '数字人视频动作和运动的描述。' }
        ]
    } },
    { id: 'ip-node-tts', tool_id: 'tts', x: 900, y: 280, status: NodeStatus.IDLE, data: { model: 'lightx2v', voiceType: 'zh_female_vv_uranus_bigtts', resourceId: 'seed-tts-2.0' } },
    { id: 'ip-node-avatar', tool_id: 'avatar-gen', x: 1500, y: 280, status: NodeStatus.IDLE, data: { model: 'SekoTalk' } }
  ]
};

export default workflow;
