import { WorkflowState, NodeStatus } from '../types';

/** @preset-id preset-s2v */
const workflow: WorkflowState = {
  id: 'preset-s2v',
  name: '文生数字人视频工作流',
      updatedAt: Date.now(),
      isDirty: false,
      isRunning: false,
      globalInputs: {},
      connections: [
        { id: 'c1', source_node_id: 'node-prompt', source_port_id: 'out-text', target_node_id: 'node-chat', target_port_id: 'in-text' },
        { id: 'c2', source_node_id: 'node-chat', source_port_id: 'image_prompt', target_node_id: 'node-image', target_port_id: 'in-text' },
        { id: 'c3', source_node_id: 'node-chat', source_port_id: 'speech_text', target_node_id: 'node-tts', target_port_id: 'in-text' },
        { id: 'c4', source_node_id: 'node-chat', source_port_id: 'tone', target_node_id: 'node-tts', target_port_id: 'in-context-tone' },
        { id: 'c5', source_node_id: 'node-image', source_port_id: 'out-image', target_node_id: 'node-avatar', target_port_id: 'in-image' },
        { id: 'c6', source_node_id: 'node-tts', source_port_id: 'out-audio', target_node_id: 'node-avatar', target_port_id: 'in-audio' },
        { id: 'c7', source_node_id: 'node-chat', source_port_id: 'avatar_video_prompt', target_node_id: 'node-avatar', target_port_id: 'in-text' }
      ],
      nodes: [
        { id: 'node-prompt', tool_id: 'text-input', x: 50, y: 200, status: NodeStatus.IDLE, data: { value: "一只哈士奇程序员狗，戴着耳机和工卡在办公，吐槽自己的程序员日常。用一些网络热梗。" } },
        { id: 'node-chat', tool_id: 'text-generation', x: 450, y: 200, status: NodeStatus.IDLE, data: {
            model: 'deepseek-v3-2-251201',
            mode: 'custom',
            customInstruction: `你是一位专业的数字人视频脚本编写者。你的任务是根据用户的输入描述，为数字人视频创建完整的脚本包。

  重要提示：
  - 生成结果中的所有字段语言必须跟随用户输入的语言。如果用户使用中文输入，所有输出字段（speech_text、tone、image_prompt、avatar_video_prompt）都必须使用中文；如果用户使用英文输入，则所有输出字段都使用英文。
  - 生成高质量、自然且引人入胜的数字人视频内容。

  对于 speech_text（语音文本）：
  - 编写自然、对话式的脚本，听起来真实可信
  - 保持简洁（正常语速下约20-40秒）
  - 使用清晰、直接的语言，符合角色的个性和风格
  - 确保脚本流畅自然，易于理解
  - 重要：不要在语音文本中使用括号、方括号或任何标记来表示语气或情感
  - 所有语气、情感和声音指令都应放在 'tone' 字段中，而不是 speech_text 中
  - 编写纯对话文本，不包含任何舞台指示或语气标记

  对于 tone（语调指令）：
  - 提供详细的配音指导，捕捉角色的个性
  - 包含情感提示、节奏、重音点和声音特征
  - 描述声音应该听起来如何（例如：温暖、权威、友好、严肃）
  - 包含语气应该转换或强调某些词语/短语的具体时刻
  - 使其对TTS系统具有可操作性，以产生自然的声音

  对于 image_prompt（肖像提示）：
  - 创建与描述角色匹配的详细肖像描述
  - 包含面部特征、年龄、表情、服装、背景、光线
  - 确保描述适合肖像图像生成
  - 匹配用户输入中角色的个性和风格
  - 包含将使头像看起来专业且引人入胜的视觉细节

  对于 avatar_video_prompt（数字人视频动作提示）：
  - 描述自然、真实的说话手势和动作
  - 包含头部动作（点头、倾斜、轻微转动）
  - 描述与语音内容和情感语调匹配的面部表情
  - 如果合适，包含肢体语言和手势
  - 指定眼神接触和视线方向
  - 确保动作与语音节奏同步
  - 足够详细，以指导数字人视频生成自然、逼真的效果

  以JSON格式输出所有四个字段。`,
            custom_outputs: [
              { id: 'speech_text', label: '语音脚本', description: '人物对听众说的话。' },
              { id: 'tone', label: '语调指令', description: '语音风格的提示。' },
              { id: 'image_prompt', label: '肖像提示', description: '用于图像生成器的人物的肖像描述。' },
              { id: 'avatar_video_prompt', label: '数字人视频动作提示', description: '数字人视频动作和运动的描述（例如：自然的说话手势、头部动作、面部表情）。' }
            ]
        } },
        { id: 'node-image', tool_id: 'text-to-image', x: 900, y: 80, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-2512', aspectRatio: "9:16" } },
        { id: 'node-tts', tool_id: 'tts', x: 900, y: 360, status: NodeStatus.IDLE, data: { model: 'lightx2v', voiceType: 'zh_male_dayi_saturn_bigtts', resourceId: 'seed-tts-2.0' } },
        { id: 'node-avatar', tool_id: 'avatar-gen', x: 1500, y: 220, status: NodeStatus.IDLE, data: {} }
      ]
  };

export default workflow;
