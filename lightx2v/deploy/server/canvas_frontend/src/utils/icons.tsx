import React from 'react';
import {
  Type, Image as ImageIcon, Volume2, Video as VideoIcon,
  Cpu, Sparkles, AlignLeft, Globe, Palette, Clapperboard,
  UserCircle, UserCog, FastForward
} from 'lucide-react';

const icons: Record<string, React.ComponentType<any>> = {
  Type,
  Image: ImageIcon,
  Volume2,
  Video: VideoIcon,
  Cpu,
  Sparkles,
  AlignLeft,
  Globe,
  Palette,
  Clapperboard,
  UserCircle,
  UserCog,
  FastForward
};

export const getIcon = (iconName: string): React.ComponentType<any> => {
  return icons[iconName] || Cpu;
};
