#!/usr/bin/env node

/**
 * 统一构建脚本
 * 自动构建 React Canvas 应用并集成到 Vue 应用中
 */

import { execSync } from 'child_process';
import { existsSync, rmSync, lstatSync, symlinkSync, readFileSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// 路径配置
// scripts 目录位置: frontend/scripts/build-with-canvas.js
const vueAppDir = join(__dirname, '..'); // frontend 目录
const serverDir = join(vueAppDir, '..'); // server 目录
const canvasAppDir = join(serverDir, 'canvas_frontend'); // canvas_frontend 目录
const canvasDistDir = join(canvasAppDir, 'dist');
const vuePublicCanvasDir = join(vueAppDir, 'public/canvas');
const vueDistDir = join(vueAppDir, 'dist');
const staticDir = join(serverDir, 'static');
const staticCanvasDir = join(staticDir, 'canvas');

console.log('路径信息:');
console.log('  Vue 应用目录:', vueAppDir);
console.log('  Canvas 应用目录:', canvasAppDir);
console.log('  Canvas 构建目录:', canvasDistDir);
console.log('  软链接目录:', vuePublicCanvasDir);
console.log('');

console.log('🚀 开始统一构建...\n');

// 1. 构建 React Canvas 应用
console.log('📦 步骤 1: 构建 React Canvas 应用...');
try {
  if (!existsSync(canvasAppDir)) {
    throw new Error(`Canvas 应用目录不存在: ${canvasAppDir}`);
  }

  // 进入 Canvas 目录并构建
  process.chdir(canvasAppDir);
  console.log(`   目录: ${canvasAppDir}`);

  // 清理旧的构建产物，确保使用最新代码
  if (existsSync(canvasDistDir)) {
    console.log('   清理旧的构建产物...');
    rmSync(canvasDistDir, { recursive: true, force: true });
    console.log('   ✅ 已清理旧的构建产物');
  }

  // 执行构建（使用 --force 确保不使用缓存）
  console.log('   执行 npm run build...');
  execSync('npm run build', {
    stdio: 'inherit',
    env: { ...process.env }
  });

  console.log('   ✅ Canvas 应用构建完成\n');
} catch (error) {
  console.error('   ❌ Canvas 应用构建失败:', error.message);
  process.exit(1);
}

// 2. 确保软链接存在
console.log('🔗 步骤 2: 检查软链接...');
try {
  const isSymlink = existsSync(vuePublicCanvasDir) &&
                    lstatSync(vuePublicCanvasDir).isSymbolicLink();

  if (!isSymlink) {
    // 如果不是软链接，删除并创建软链接
    if (existsSync(vuePublicCanvasDir)) {
      rmSync(vuePublicCanvasDir, { recursive: true, force: true });
    }
    // 创建软链接
    symlinkSync('../../canvas_frontend/dist', vuePublicCanvasDir, 'dir');
    console.log(`   ✅ 已创建软链接: ${vuePublicCanvasDir} -> ../../canvas_frontend/dist\n`);
  } else {
    console.log(`   ✅ 软链接已存在: ${vuePublicCanvasDir}\n`);
  }
} catch (error) {
  console.error('   ❌ 软链接检查/创建失败:', error.message);
  process.exit(1);
}

// 3. 验证构建产物
console.log('✅ 步骤 3: 验证构建产物...');
try {
  if (!existsSync(canvasDistDir)) {
    throw new Error(`Canvas 构建产物不存在: ${canvasDistDir}`);
  }
  if (!existsSync(join(canvasDistDir, 'index.html'))) {
    throw new Error(`Canvas index.html 不存在`);
  }
  console.log(`   ✅ 构建产物验证通过\n`);
} catch (error) {
  console.error('   ❌ 验证失败:', error.message);
  process.exit(1);
}

// 3.5. 修复 Canvas HTML 文件（添加 type="module" 到包含 import() 的脚本，并确保初始化脚本在正确位置）
console.log('🔧 步骤 3.5: 修复 Canvas HTML 文件...');
try {
  const canvasHtmlPath = join(canvasDistDir, 'index.html');
  let html = readFileSync(canvasHtmlPath, 'utf-8');

  let modifiedHtml = html;

  // 1. 确保 body 中的生命周期脚本有 entry 标记，并且直接导出生命周期函数到全局作用域
  // qiankun 需要找到这些函数，所以必须确保它们被正确导出
  modifiedHtml = modifiedHtml.replace(
    /<script>(\s*const createDeffer = \(hookName\) => {[\s\S]*?global\['react-canvas'\] = \{[\s\S]*?\}[\s\S]*?<\/script>)\s*(?=<\/body>)/g,
    (match) => {
      // 在脚本开头添加 window.proxy 初始化，确保 createDeffer 能正常工作
      let modified = match.replace(
        /^(<script>)/,
        `<script entry>
  // 确保 window.proxy 存在（必须在 createDeffer 之前）
  if (!window.proxy) {
    window.proxy = {};
  }
  if (!window.moudleQiankunAppLifeCycles) {
    window.moudleQiankunAppLifeCycles = {};
  }
  `
      );
      // 确保在脚本末尾添加全局导出
      // qiankun 需要从全局作用域中找到这些函数
      modified = modified.replace(
        /(global\['react-canvas'\] = \{[\s\S]*?\};)/,
        `$1
    // 直接导出到全局作用域，以便 qiankun 能在 entry 脚本中找到
    // qiankun 会在执行 entry 脚本后立即检查这些函数是否存在
    window.bootstrap = bootstrap;
    window.mount = mount;
    window.unmount = unmount;
    window.update = update;`
      );
      return modified;
    }
  );

  // 2. 确保在 head 中有初始化脚本（在所有其他脚本之前）
  if (!modifiedHtml.includes('global.qiankunName = \'react-canvas\';')) {
    // 在 head 标签开始后插入初始化脚本
    modifiedHtml = modifiedHtml.replace(
      /(<head[^>]*>)/,
      (match) => {
        return match + `\n<!-- 初始化全局对象，必须在所有其他脚本之前 -->\n<script entry>\n  // 确保 window.proxy 和 moudleQiankunAppLifeCycles 在 vite-plugin-qiankun 使用前就存在\n  if (!window.proxy) {\n    window.proxy = {}\n  }\n  if (!window.moudleQiankunAppLifeCycles) {\n    window.moudleQiankunAppLifeCycles = {}\n  }\n\n  // 创建延迟函数，用于 qiankun 生命周期\n  const createDeffer = (hookName) => {\n    const d = new Promise((resolve, reject) => {\n      window.proxy && (window.proxy[\`vite\${hookName}\`] = resolve)\n    })\n    return props => d.then(fn => fn(props));\n  }\n\n  // 注册 qiankun 生命周期（使用延迟执行，等待模块加载完成后连接）\n  ;(global => {\n    global.qiankunName = 'react-canvas';\n    global['react-canvas'] = {\n      bootstrap: createDeffer('bootstrap'),\n      mount: createDeffer('mount'),\n      unmount: createDeffer('unmount'),\n      update: createDeffer('update')\n    };\n  })(window);\n</script>`;
      }
    );
  }

  // 3. 确保模块脚本有 type="module" 属性，并修复路径
  // 同时添加 data-qiankun-ignore 属性，防止 qiankun 将其作为 entry 脚本执行
  modifiedHtml = modifiedHtml.replace(
    /<script([^>]*?)>([\s\S]*?import\([^)]+\)[\s\S]*?)<\/script>/g,
    (match, attrs, content) => {
      // 修复相对路径为绝对路径
      let fixedContent = content.replace(
        /import\(['"](\.\/assets\/[^'"]+)['"]\)/g,
        (match, path) => `import('/canvas${path.replace(/^\./, '')}')`
      );

      // 确保所有 /assets/ 路径都是 /canvas/assets/
      fixedContent = fixedContent.replace(
        /import\(['"](\/assets\/[^'"]+)['"]\)/g,
        (match, path) => {
          if (path.startsWith('/canvas/')) {
            return match;
          }
          return `import('/canvas${path}')`;
        }
      );

      // 添加 type="module" 和数据属性
      let newAttrs = attrs;
      if (!newAttrs.includes('type="module"') && !newAttrs.includes("type='module'")) {
        newAttrs = ` type="module"${newAttrs}`;
      }
      // 添加 ignore 属性，防止 qiankun 将其识别为 entry 脚本
      if (!newAttrs.includes('data-qiankun-ignore') && !newAttrs.includes("data-qiankun-ignore")) {
        newAttrs = `${newAttrs} data-qiankun-ignore`;
      }

      // 如果有修改，返回新的脚本标签
      if (content !== fixedContent || attrs !== newAttrs) {
        return `<script${newAttrs}>${fixedContent}</script>`;
      }
      return match;
    }
  );

  if (html !== modifiedHtml) {
    writeFileSync(canvasHtmlPath, modifiedHtml, 'utf-8');
    console.log('   ✅ 已修复 HTML 文件：添加 qiankun 初始化脚本和 type="module" 属性\n');
  } else {
    console.log('   ✅ HTML 文件已正确配置\n');
  }
} catch (error) {
  console.error('   ⚠️  修复 HTML 失败:', error.message);
  // 非致命错误，继续执行
}

// 4. 构建 Vue 应用
console.log('📦 步骤 4: 构建 Vue 主应用...');
try {
  process.chdir(vueAppDir);
  console.log(`   目录: ${vueAppDir}`);

  // 清理旧的构建产物，确保使用最新代码
  if (existsSync(vueDistDir)) {
    console.log('   清理旧的构建产物...');
    rmSync(vueDistDir, { recursive: true, force: true });
    console.log('   ✅ 已清理旧的构建产物');
  }

  console.log('   执行 npm run build...');
  execSync('npm run build', {
    stdio: 'inherit',
    env: { ...process.env }
  });

  console.log('   ✅ Vue 应用构建完成\n');
} catch (error) {
  console.error('   ❌ Vue 应用构建失败:', error.message);
  process.exit(1);
}

// 5. 确保 static 目录下的软链接正确指向 dist 目录
console.log('🔗 步骤 5: 设置 static 目录软链接...');
try {
  // 确保 static 目录存在
  if (!existsSync(staticDir)) {
    console.error(`   ❌ static 目录不存在: ${staticDir}`);
    process.exit(1);
  }

  // 设置主应用的 index.html 软链接
  const staticIndexHtml = join(staticDir, 'index.html');
  if (existsSync(staticIndexHtml)) {
    if (lstatSync(staticIndexHtml).isSymbolicLink()) {
      rmSync(staticIndexHtml);
    } else {
      rmSync(staticIndexHtml, { force: true });
    }
  }
  symlinkSync(join(vueDistDir, 'index.html'), staticIndexHtml, 'file');
  console.log(`   ✅ 已设置软链接: ${staticIndexHtml} -> ${join(vueDistDir, 'index.html')}`);

  // 设置主应用的 assets 软链接
  const staticAssetsDir = join(staticDir, 'assets');
  if (existsSync(staticAssetsDir)) {
    if (lstatSync(staticAssetsDir).isSymbolicLink()) {
      rmSync(staticAssetsDir);
    } else {
      rmSync(staticAssetsDir, { recursive: true, force: true });
    }
  }
  symlinkSync(join(vueDistDir, 'assets'), staticAssetsDir, 'dir');
  console.log(`   ✅ 已设置软链接: ${staticAssetsDir} -> ${join(vueDistDir, 'assets')}`);

  // 设置 Canvas 应用的软链接
  if (existsSync(staticCanvasDir)) {
    if (lstatSync(staticCanvasDir).isSymbolicLink()) {
      rmSync(staticCanvasDir);
    } else {
      rmSync(staticCanvasDir, { recursive: true, force: true });
    }
  }
  symlinkSync(join(vueDistDir, 'canvas'), staticCanvasDir, 'dir');
  console.log(`   ✅ 已设置软链接: ${staticCanvasDir} -> ${join(vueDistDir, 'canvas')}\n`);
} catch (error) {
  console.error('   ❌ 软链接设置失败:', error.message);
  process.exit(1);
}

console.log('🎉 统一构建完成！');
console.log(`\n📁 构建产物位置: ${join(vueAppDir, 'dist')}`);
console.log(`📁 static 目录已同步: ${staticDir}`);
console.log('💡 现在可以通过 static 目录部署整个应用了\n');
