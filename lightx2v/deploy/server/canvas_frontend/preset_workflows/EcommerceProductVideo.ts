import { WorkflowState, NodeStatus } from '../types';

/** @preset-id preset-ecommerce-product-video */
const workflow: WorkflowState = {
  id: 'preset-ecommerce-product-video',
  name: '电商多视角产品精修',
  updatedAt: Date.now(),
  isDirty: false,
  isRunning: false,
  globalInputs: {},
  connections: [
    { id: 'ec-c1', source_node_id: 'ec-node-product', source_port_id: 'out-image', target_node_id: 'ec-node-llm', target_port_id: 'in-image' },
    { id: 'ec-c2', source_node_id: 'ec-node-refinement-hint', source_port_id: 'out-text', target_node_id: 'ec-node-llm', target_port_id: 'in-text' },
    { id: 'ec-c3', source_node_id: 'ec-node-product', source_port_id: 'out-image', target_node_id: 'ec-node-i2i-front', target_port_id: 'in-image' },
    { id: 'ec-c4', source_node_id: 'ec-node-llm', source_port_id: 'refinement_front', target_node_id: 'ec-node-i2i-front', target_port_id: 'in-text' },
    { id: 'ec-c5', source_node_id: 'ec-node-product', source_port_id: 'out-image', target_node_id: 'ec-node-i2i-side', target_port_id: 'in-image' },
    { id: 'ec-c6', source_node_id: 'ec-node-llm', source_port_id: 'refinement_side', target_node_id: 'ec-node-i2i-side', target_port_id: 'in-text' },
    { id: 'ec-c7', source_node_id: 'ec-node-product', source_port_id: 'out-image', target_node_id: 'ec-node-i2i-back', target_port_id: 'in-image' },
    { id: 'ec-c8', source_node_id: 'ec-node-llm', source_port_id: 'refinement_back', target_node_id: 'ec-node-i2i-back', target_port_id: 'in-text' },
    { id: 'ec-c9', source_node_id: 'ec-node-product', source_port_id: 'out-image', target_node_id: 'ec-node-i2i-top', target_port_id: 'in-image' },
    { id: 'ec-c10', source_node_id: 'ec-node-llm', source_port_id: 'refinement_top', target_node_id: 'ec-node-i2i-top', target_port_id: 'in-text' },
    { id: 'ec-c11', source_node_id: 'ec-node-i2i-front', source_port_id: 'out-image', target_node_id: 'ec-node-vid-front', target_port_id: 'in-image' },
    { id: 'ec-c12', source_node_id: 'ec-node-llm', source_port_id: 'motion_front', target_node_id: 'ec-node-vid-front', target_port_id: 'in-text' },
    { id: 'ec-c13', source_node_id: 'ec-node-i2i-side', source_port_id: 'out-image', target_node_id: 'ec-node-vid-side', target_port_id: 'in-image' },
    { id: 'ec-c14', source_node_id: 'ec-node-llm', source_port_id: 'motion_side', target_node_id: 'ec-node-vid-side', target_port_id: 'in-text' },
    { id: 'ec-c15', source_node_id: 'ec-node-i2i-back', source_port_id: 'out-image', target_node_id: 'ec-node-vid-back', target_port_id: 'in-image' },
    { id: 'ec-c16', source_node_id: 'ec-node-llm', source_port_id: 'motion_back', target_node_id: 'ec-node-vid-back', target_port_id: 'in-text' },
    { id: 'ec-c17', source_node_id: 'ec-node-i2i-top', source_port_id: 'out-image', target_node_id: 'ec-node-vid-top', target_port_id: 'in-image' },
    { id: 'ec-c18', source_node_id: 'ec-node-llm', source_port_id: 'motion_top', target_node_id: 'ec-node-vid-top', target_port_id: 'in-text' }
  ],
  nodes: [
    {
      id: 'ec-node-product',
      tool_id: 'image-input',
      x: 50,
      y: 80,
      status: NodeStatus.IDLE,
      data: { value: ['/assets/shoe.jpg'] }
    },
    {
      id: 'ec-node-refinement-hint',
      tool_id: 'text-input',
      x: 50,
      y: 360,
      status: NodeStatus.IDLE,
      data: {
        value: '产品置于纯净纯白背景、正视图/平视、3D 渲染感、精准还原颜色与材质（玻璃通透/塑料哑光/金属光泽等）、清除指纹灰尘瑕疵、光影立体、标签文字清晰、光线柔和均匀'
      }
    },
    {
      id: 'ec-node-llm',
      tool_id: 'text-generation',
      x: 280,
      y: 120,
      status: NodeStatus.IDLE,
      data: {
        model: 'doubao-seed-1-6-vision-250815',
        mode: 'custom',
        customInstruction: `你是一位电商主图与多视角精修、动效运镜专家。用户会提供一张产品图（in-image）和一段精修要求（in-text，可修改），请结合两者生成八个字段，语言与用户输入一致（默认中文）。每个字段只输出该段文本，不要加字段名或前缀。

精修提示词（用于图生图）：
refinement_front：以 in-text 为基础，描述产品正面视角的精修效果（正视图、平视、主视觉面）。
refinement_side：描述产品侧面视角的精修效果（如 45° 侧视、侧面质感与光影）。
refinement_back：描述产品背面视角的精修效果（背面细节、标签、风格一致）。
refinement_top：描述产品俯视或局部细节的精修效果（顶视图、细节特写、材质纹理）。

运镜/动效提示词（用于图生视频，强调镜头运动与画面动效）：
motion_front：正面精修图的动效与运镜，如缓慢推进、轻微旋转、光影流动、产品高光闪烁等。
motion_side：侧面视角的运镜与动效，如绕产品缓慢旋转、侧面光影变化、质感凸显等。
motion_back：背面视角的运镜与动效，如从背面向正面过渡、背面细节渐显等。
motion_top：俯视/细节的运镜与动效，如俯拍缓慢下移、细节特写推镜、材质纹理微动等。`,
        custom_outputs: [
          { id: 'refinement_front', label: '正面精修', description: '正面视角图生图精修提示' },
          { id: 'refinement_side', label: '侧面精修', description: '侧面视角图生图精修提示' },
          { id: 'refinement_back', label: '背面精修', description: '背面视角图生图精修提示' },
          { id: 'refinement_top', label: '俯视/细节精修', description: '俯视或细节图生图精修提示' },
          { id: 'motion_front', label: '正面运镜动效', description: '正面图生视频的运镜与动效' },
          { id: 'motion_side', label: '侧面运镜动效', description: '侧面图生视频的运镜与动效' },
          { id: 'motion_back', label: '背面运镜动效', description: '背面图生视频的运镜与动效' },
          { id: 'motion_top', label: '俯视运镜动效', description: '俯视图生视频的运镜与动效' }
        ]
      }
    },
    {
      id: 'ec-node-i2i-front',
      tool_id: 'image-to-image',
      x: 560,
      y: 80,
      status: NodeStatus.IDLE,
      data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '3:4' }
    },
    {
      id: 'ec-node-i2i-side',
      tool_id: 'image-to-image',
      x: 560,
      y: 200,
      status: NodeStatus.IDLE,
      data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '3:4' }
    },
    {
      id: 'ec-node-i2i-back',
      tool_id: 'image-to-image',
      x: 560,
      y: 320,
      status: NodeStatus.IDLE,
      data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '3:4' }
    },
    {
      id: 'ec-node-i2i-top',
      tool_id: 'image-to-image',
      x: 560,
      y: 440,
      status: NodeStatus.IDLE,
      data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '3:4' }
    },
    {
      id: 'ec-node-vid-front',
      tool_id: 'video-gen-image',
      x: 900,
      y: 80,
      status: NodeStatus.IDLE,
      data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '3:4' }
    },
    {
      id: 'ec-node-vid-side',
      tool_id: 'video-gen-image',
      x: 900,
      y: 200,
      status: NodeStatus.IDLE,
      data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '3:4' }
    },
    {
      id: 'ec-node-vid-back',
      tool_id: 'video-gen-image',
      x: 900,
      y: 320,
      status: NodeStatus.IDLE,
      data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '3:4' }
    },
    {
      id: 'ec-node-vid-top',
      tool_id: 'video-gen-image',
      x: 900,
      y: 440,
      status: NodeStatus.IDLE,
      data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '3:4' }
    }
  ]
};

export default workflow;
