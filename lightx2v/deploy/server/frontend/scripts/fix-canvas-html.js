#!/usr/bin/env node

/**
 * 修复 Canvas HTML 文件
 * 给包含 import() 的脚本添加 type="module"，以便 qiankun 正确加载
 */

import { readFileSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const canvasHtmlPath = join(__dirname, '../public/canvas/index.html');

try {
  let html = readFileSync(canvasHtmlPath, 'utf-8');
  let modified = false;

  // 查找包含 import() 的脚本标签，修复路径和添加 type="module"
  let modifiedHtml = html.replace(
    /<script([^>]*?)>([\s\S]*?import\([^)]+\)[\s\S]*?)<\/script>/g,
    (match, attrs, content) => {
      // 修复相对路径为绝对路径（相对于 /canvas/）
      let fixedContent = content.replace(
        /import\(['"](\.\/assets\/[^'"]+)['"]\)/g,
        (match, path) => {
          const absolutePath = `/canvas${path.replace(/^\./, '')}`;
          if (path !== absolutePath) {
            modified = true;
            return `import('${absolutePath}')`;
          }
          return match;
        }
      );

      // 如果已经有 type="module"，只修复路径
      if (attrs.includes('type="module"') || attrs.includes("type='module'")) {
        if (content !== fixedContent) {
          modified = true;
          return `<script${attrs}>${fixedContent}</script>`;
        }
        return match;
      }
      // 添加 type="module" 属性并修复路径
      modified = true;
      return `<script type="module"${attrs}>${fixedContent}</script>`;
    }
  );

  if (modified) {
    writeFileSync(canvasHtmlPath, modifiedHtml, 'utf-8');
    console.log('✅ 已修复 Canvas HTML 文件：添加 type="module" 并修复资源路径');
  } else {
    console.log('✅ Canvas HTML 文件已正确配置');
  }
} catch (error) {
  console.error('❌ 修复 HTML 文件失败:', error.message);
  process.exit(1);
}
