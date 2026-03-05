import { WorkflowState, NodeStatus } from '../types';

/** @preset-id preset-morph */
const workflow: WorkflowState = {
  id: 'preset-morph',
  name: '文生首尾帧视频工作流',
      updatedAt: Date.now(),
      isDirty: false,
      isRunning: false,
      globalInputs: {},
      connections: [
        { id: 'c1', source_node_id: 'node-input', source_port_id: 'out-text', target_node_id: 'node-planner', target_port_id: 'in-text' },
        { id: 'c2', source_node_id: 'node-planner', source_port_id: 'start_img_prompt', target_node_id: 'node-start-frame', target_port_id: 'in-text' },
        { id: 'c3', source_node_id: 'node-start-frame', source_port_id: 'out-image', target_node_id: 'node-end-frame', target_port_id: 'in-image' },
        { id: 'c4', source_node_id: 'node-planner', source_port_id: 'end_img_prompt', target_node_id: 'node-end-frame', target_port_id: 'in-text' },
        { id: 'c5', source_node_id: 'node-start-frame', source_port_id: 'out-image', target_node_id: 'node-video', target_port_id: 'in-image-start' },
        { id: 'c6', source_node_id: 'node-end-frame', source_port_id: 'out-image', target_node_id: 'node-video', target_port_id: 'in-image-end' },
        { id: 'c7', source_node_id: 'node-planner', source_port_id: 'video_motion_prompt', target_node_id: 'node-video', target_port_id: 'in-text' }
      ],
      nodes: [
        { id: 'node-input', tool_id: 'text-input', x: 50, y: 300, status: NodeStatus.IDLE, data: { value: "一座未来主义赛博朋克城市，从白天逐渐过渡到雨夜。" } },
        { id: 'node-planner', tool_id: 'text-generation', x: 450, y: 300, status: NodeStatus.IDLE, data: {
            model: 'deepseek-v3-2-251201',
            mode: 'custom',
            customInstruction: `You are a video planning assistant. Analyze the input description and generate detailed prompts for the start frame, end frame, and video motion.

IMPORTANT: All output fields must use the same language as the user's input. If the user inputs in Chinese, all output fields (start_img_prompt, end_img_prompt, video_motion_prompt) must be in Chinese. If the user inputs in English, all output fields must be in English.

Generate:
- start_img_prompt: Detailed prompt for the initial image
- end_img_prompt: Detailed prompt for the target image, based on the start
- video_motion_prompt: Prompt describing the transition and camera motion`,
            custom_outputs: [
              { id: 'start_img_prompt', label: 'Start Frame Prompt', description: 'Detailed prompt for the initial image.' },
              { id: 'end_img_prompt', label: 'End Frame Prompt', description: 'Detailed prompt for the target image, based on the start.' },
              { id: 'video_motion_prompt', label: 'Motion Prompt', description: 'Prompt describing the transition and camera motion.' }
            ]
        } },
        { id: 'node-start-frame', tool_id: 'text-to-image', x: 900, y: 80, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-2512', aspectRatio: "16:9" } },
        { id: 'node-end-frame', tool_id: 'image-to-image', x: 900, y: 440, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511' } },
        { id: 'node-video', tool_id: 'video-gen-dual-frame', x: 1500, y: 260, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: "16:9" } }
      ]
    };

export default workflow;
