import { WorkflowState, NodeStatus } from '../types';

/** @preset-id preset-product-wearing */
const workflow: WorkflowState = {
  id: 'preset-product-wearing',
  name: '虚拟人结合电商产品工作流',
    updatedAt: Date.now(),
    isDirty: false,
    isRunning: false,
    globalInputs: {},
    connections: [
      // Input to AI Chat
      { id: 'prod-c1', source_node_id: 'prod-node-person', source_port_id: 'out-image', target_node_id: 'prod-node-planner', target_port_id: 'in-image' },
      { id: 'prod-c2', source_node_id: 'prod-node-product', source_port_id: 'out-image', target_node_id: 'prod-node-planner', target_port_id: 'in-image' },
      { id: 'prod-c3', source_node_id: 'prod-node-text', source_port_id: 'out-text', target_node_id: 'prod-node-planner', target_port_id: 'in-text' },
      // AI Chat to first image-to-image (front view) - both person and product images
      { id: 'prod-c4', source_node_id: 'prod-node-planner', source_port_id: 'front_prompt', target_node_id: 'prod-node-i2i-front', target_port_id: 'in-text' },
      { id: 'prod-c5', source_node_id: 'prod-node-person', source_port_id: 'out-image', target_node_id: 'prod-node-i2i-front', target_port_id: 'in-image' },
      { id: 'prod-c5b', source_node_id: 'prod-node-product', source_port_id: 'out-image', target_node_id: 'prod-node-i2i-front', target_port_id: 'in-image' },
      // All subsequent i2i nodes use the front image as base
      { id: 'prod-c6', source_node_id: 'prod-node-i2i-front', source_port_id: 'out-image', target_node_id: 'prod-node-i2i-right45', target_port_id: 'in-image' },
      { id: 'prod-c7', source_node_id: 'prod-node-planner', source_port_id: 'right45_prompt', target_node_id: 'prod-node-i2i-right45', target_port_id: 'in-text' },
      { id: 'prod-c8', source_node_id: 'prod-node-i2i-front', source_port_id: 'out-image', target_node_id: 'prod-node-i2i-side90', target_port_id: 'in-image' },
      { id: 'prod-c9', source_node_id: 'prod-node-planner', source_port_id: 'side90_prompt', target_node_id: 'prod-node-i2i-side90', target_port_id: 'in-text' },
      { id: 'prod-c10', source_node_id: 'prod-node-i2i-front', source_port_id: 'out-image', target_node_id: 'prod-node-i2i-left45', target_port_id: 'in-image' },
      { id: 'prod-c11', source_node_id: 'prod-node-planner', source_port_id: 'left45_prompt', target_node_id: 'prod-node-i2i-left45', target_port_id: 'in-text' },
      { id: 'prod-c12', source_node_id: 'prod-node-i2i-front', source_port_id: 'out-image', target_node_id: 'prod-node-i2i-low', target_port_id: 'in-image' },
      { id: 'prod-c13', source_node_id: 'prod-node-planner', source_port_id: 'low_prompt', target_node_id: 'prod-node-i2i-low', target_port_id: 'in-text' },
      { id: 'prod-c14', source_node_id: 'prod-node-i2i-front', source_port_id: 'out-image', target_node_id: 'prod-node-i2i-high', target_port_id: 'in-image' },
      { id: 'prod-c15', source_node_id: 'prod-node-planner', source_port_id: 'high_prompt', target_node_id: 'prod-node-i2i-high', target_port_id: 'in-text' },
      // All images to video generation
      { id: 'prod-c16', source_node_id: 'prod-node-i2i-front', source_port_id: 'out-image', target_node_id: 'prod-node-video-front', target_port_id: 'in-image' },
      { id: 'prod-c17', source_node_id: 'prod-node-planner', source_port_id: 'front_motion', target_node_id: 'prod-node-video-front', target_port_id: 'in-text' },
      { id: 'prod-c18', source_node_id: 'prod-node-i2i-right45', source_port_id: 'out-image', target_node_id: 'prod-node-video-right45', target_port_id: 'in-image' },
      { id: 'prod-c19', source_node_id: 'prod-node-planner', source_port_id: 'right45_motion', target_node_id: 'prod-node-video-right45', target_port_id: 'in-text' },
      { id: 'prod-c20', source_node_id: 'prod-node-i2i-side90', source_port_id: 'out-image', target_node_id: 'prod-node-video-side90', target_port_id: 'in-image' },
      { id: 'prod-c21', source_node_id: 'prod-node-planner', source_port_id: 'side90_motion', target_node_id: 'prod-node-video-side90', target_port_id: 'in-text' },
      { id: 'prod-c22', source_node_id: 'prod-node-i2i-left45', source_port_id: 'out-image', target_node_id: 'prod-node-video-left45', target_port_id: 'in-image' },
      { id: 'prod-c23', source_node_id: 'prod-node-planner', source_port_id: 'left45_motion', target_node_id: 'prod-node-video-left45', target_port_id: 'in-text' },
      { id: 'prod-c24', source_node_id: 'prod-node-i2i-low', source_port_id: 'out-image', target_node_id: 'prod-node-video-low', target_port_id: 'in-image' },
      { id: 'prod-c25', source_node_id: 'prod-node-planner', source_port_id: 'low_motion', target_node_id: 'prod-node-video-low', target_port_id: 'in-text' },
      { id: 'prod-c26', source_node_id: 'prod-node-i2i-high', source_port_id: 'out-image', target_node_id: 'prod-node-video-high', target_port_id: 'in-image' },
      { id: 'prod-c27', source_node_id: 'prod-node-planner', source_port_id: 'high_motion', target_node_id: 'prod-node-video-high', target_port_id: 'in-text' }
    ],
    nodes: [
      // Input nodes (vertical spacing 280px to avoid overlap with taller image/audio/video input nodes)
      { id: 'prod-node-person', tool_id: 'image-input', x: 50, y: 80, status: NodeStatus.IDLE, data: { value: ['/assets/model_girl.png'] } },
      { id: 'prod-node-product', tool_id: 'image-input', x: 50, y: 360, status: NodeStatus.IDLE, data: { value: ['/assets/product_glass.png'] } },
      { id: 'prod-node-text', tool_id: 'text-input', x: 50, y: 640, status: NodeStatus.IDLE, data: { value: "" } },
      // AI Chat Planner (Doubao Vision)
      { id: 'prod-node-planner', tool_id: 'text-generation', x: 450, y: 280, status: NodeStatus.IDLE, data: {
          model: 'doubao-seed-1-6-vision-250815',
          mode: 'custom',
          customInstruction: `你是一位专业的虚拟人产品展示创意总监。你的任务是根据输入的人物图片、产品图片和可选的文字描述，生成详细的图生图修改提示词和视频运镜提示词。

重要原则：
- 生成结果中的所有字段语言必须跟随用户输入的语言。如果用户使用中文输入，所有输出字段（front_prompt、front_motion、right45_prompt等）都必须使用中文；如果用户使用英文输入，则所有输出字段都使用英文。
- 保持人物特征的一致性（面部特征、身材、服装等）
- 产品必须准确、自然地佩戴在人物身上
- 每个角度的图片都要保持产品佩戴位置和方式的准确性
- 确保所有角度的图片风格统一

你需要生成6个角度的图生图修改提示词和对应的视频运镜提示词：

1. 正面图（front_prompt）：让输入的人物图片中的人物佩戴输入的产品图片中的产品，保持人物和产品的一致性
2. 右边45度侧面（right45_prompt）：将正面图中的人物向右转45度，微微低头，保持人物和产品的一致性
3. 90度完全侧面（side90_prompt）：将正面图中的人物完全侧身，保持人物和产品的一致性
4. 左边45度侧面（left45_prompt）：将正面图中的人物向左转45度，保持人物和产品的一致性
5. 仰拍角度（low_prompt）：将正面图中的人物改为仰拍角度，从下往上拍摄，保持人物和产品的一致性
6. 俯拍角度（high_prompt）：将正面图中的人物改为俯拍角度，从上往下拍摄，保持人物和产品的一致性

每个图生图修改提示词应该：
- 强调"改"和"一致性"，不需要描述图片本身的内容
- 明确说明如何修改：例如"让图一的人物佩戴图二的眼镜，保持人物和物品一致性"
- 描述需要改变的姿势、角度、视角
- 强调保持人物特征和产品的一致性
- 不要描述光线、背景、氛围等图片细节

每个视频运镜提示词应该：
- 描述简单的相机运动，如"镜头从下移动至上，聚焦模特"
- 与对应角度的图片内容匹配
- 简洁明了，适合图生视频使用

输出格式：JSON，包含以下字段：
- front_prompt, front_motion
- right45_prompt, right45_motion
- side90_prompt, side90_motion
- left45_prompt, left45_motion
- low_prompt, low_motion
- high_prompt, high_motion`,
          custom_outputs: [
            { id: 'front_prompt', label: '正面图提示词', description: '人物正面佩戴产品的详细图生图提示词' },
            { id: 'front_motion', label: '正面运镜提示词', description: '正面图的视频运镜提示词' },
            { id: 'right45_prompt', label: '右边45度提示词', description: '右边45度侧面，人物微微低头的图生图提示词' },
            { id: 'right45_motion', label: '右边45度运镜提示词', description: '右边45度图的视频运镜提示词' },
            { id: 'side90_prompt', label: '90度侧面提示词', description: '完全侧面的图生图提示词' },
            { id: 'side90_motion', label: '90度侧面运镜提示词', description: '90度侧面图的视频运镜提示词' },
            { id: 'left45_prompt', label: '左边45度提示词', description: '左边45度侧面的图生图提示词' },
            { id: 'left45_motion', label: '左边45度运镜提示词', description: '左边45度图的视频运镜提示词' },
            { id: 'low_prompt', label: '仰拍提示词', description: '仰拍角度的图生图提示词' },
            { id: 'low_motion', label: '仰拍运镜提示词', description: '仰拍图的视频运镜提示词' },
            { id: 'high_prompt', label: '俯拍提示词', description: '俯拍角度的图生图提示词' },
            { id: 'high_motion', label: '俯拍运镜提示词', description: '俯拍图的视频运镜提示词' }
          ]
      } },
      // Image-to-image nodes (6 angles)
      { id: 'prod-node-i2i-front', tool_id: 'image-to-image', x: 900, y: 50, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'prod-node-i2i-right45', tool_id: 'image-to-image', x: 900, y: 150, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'prod-node-i2i-side90', tool_id: 'image-to-image', x: 900, y: 250, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'prod-node-i2i-left45', tool_id: 'image-to-image', x: 900, y: 350, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'prod-node-i2i-low', tool_id: 'image-to-image', x: 900, y: 450, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      { id: 'prod-node-i2i-high', tool_id: 'image-to-image', x: 900, y: 550, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
      // Video generation nodes (6 videos)
      { id: 'prod-node-video-front', tool_id: 'video-gen-image', x: 1500, y: 50, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
      { id: 'prod-node-video-right45', tool_id: 'video-gen-image', x: 1500, y: 150, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
      { id: 'prod-node-video-side90', tool_id: 'video-gen-image', x: 1500, y: 250, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
      { id: 'prod-node-video-left45', tool_id: 'video-gen-image', x: 1500, y: 350, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
      { id: 'prod-node-video-low', tool_id: 'video-gen-image', x: 1500, y: 450, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
      { id: 'prod-node-video-high', tool_id: 'video-gen-image', x: 1500, y: 550, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } }
    ]
  };

export default workflow;
