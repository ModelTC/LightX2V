
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { initLightX2VToken, clearTokenCache } from './src/utils/apiClient';

// 声明全局类型
declare global {
  interface Window {
    __POWERED_BY_QIANKUN__?: boolean;
    __INJECTED_PUBLIC_PATH_BY_QIANKUN__?: string;
  }
}

// qiankun 生命周期函数
let root: any = null;

/**
 * bootstrap 只会在微应用初始化的时候调用一次，下次微应用重新进入时会直接调用 mount 钩子，不会再重复触发 bootstrap。
 */
export async function bootstrap() {
  console.log('[React] Canvas app bootstrapped');
}

/**
 * 应用每次进入都会调用 mount 方法，通常我们在这里触发应用的渲染方法
 */
export async function mount(props: any) {
  console.log('[React] Canvas app mounted', props);

  // 设置资源基础路径
  (window as any).__ASSET_BASE_PATH__ = '/canvas';

  // 将共享的状态和方法挂载到 window，供 App 组件使用
  if (props?.sharedStore) {
    (window as any).__SHARED_STORE__ = props.sharedStore;
    const store = props.sharedStore;
    // 优先用主应用传下来的 token，否则从 localStorage 取（主应用登录后可能只写了 localStorage 未同步到 sharedStore）
    const token = store.getState?.('token');
    const initialState = typeof props.getInitialState === 'function' ? props.getInitialState() : null;
    let tokenToSync = token ?? initialState?.token;
    if (typeof localStorage !== 'undefined') {
      const localToken = localStorage.getItem('accessToken');
      if (tokenToSync) {
        localStorage.setItem('accessToken', tokenToSync);
      } else if (localToken) {
        // 主应用未同步到 sharedStore 时，从 localStorage 回填，保证首屏请求带 token
        store.setState?.('token', localToken);
        tokenToSync = localToken;
      }
    }
  }
  if (props?.apiClient) {
    (window as any).__API_CLIENT__ = props.apiClient;
  }
  if (props?.setGlobalState) {
    (window as any).__SET_GLOBAL_STATE__ = props.setGlobalState;
  }
  if (props?.onGlobalStateChange) {
    (window as any).__ON_GLOBAL_STATE_CHANGE__ = props.onGlobalStateChange;
  }

  clearTokenCache();
  initLightX2VToken();

  // 检查环境变量（用于调试）
  console.log('[Canvas App] 环境变量检查:', {
    DEEPSEEK_API_KEY: process.env.DEEPSEEK_API_KEY ? `${process.env.DEEPSEEK_API_KEY.substring(0, 10)}...` : '未设置',
    GEMINI_API_KEY: process.env.GEMINI_API_KEY ? `${process.env.GEMINI_API_KEY.substring(0, 10)}...` : '未设置',
    PPCHAT_API_KEY: process.env.PPCHAT_API_KEY ? `${process.env.PPCHAT_API_KEY.substring(0, 10)}...` : '未设置',
    LIGHTX2V_URL: process.env.LIGHTX2V_URL || '未设置',
    LIGHTX2V_CLOUD_URL: process.env.LIGHTX2V_CLOUD_URL || '未设置',
    LIGHTX2V_CLOUD_TOKEN: process.env.LIGHTX2V_CLOUD_TOKEN ? `${process.env.LIGHTX2V_CLOUD_TOKEN}` : '未设置',
  });

  const rootElement = props?.container
    ? props.container.querySelector('#root')
    : document.getElementById('root');

  if (!rootElement) {
    throw new Error("Could not find root element to mount to");
  }

  root = ReactDOM.createRoot(rootElement);
  root.render(
    <React.StrictMode>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </React.StrictMode>
  );
}

/**
 * 应用每次 切出/卸载 会调用的方法，通常在这里我们会卸载微应用的应用实例
 */
export async function unmount(props: any) {
  console.log('[React] Canvas app unmounted', props);
  if (root) {
    root.unmount();
    root = null;
  }

  // 清理全局变量
  delete (window as any).__SHARED_STORE__;
  delete (window as any).__API_CLIENT__;
  delete (window as any).__SET_GLOBAL_STATE__;
  delete (window as any).__ON_GLOBAL_STATE_CHANGE__;
}

// Error Boundary Component
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: Error | null; errorInfo: React.ErrorInfo | null }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
    this.setState({ error, errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: '#0f172a',
          color: '#f1f5f9',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '2rem',
          fontFamily: 'system-ui, -apple-system, sans-serif',
          zIndex: 9999
        }}>
          <div style={{
            maxWidth: '600px',
            width: '100%',
            backgroundColor: '#1e293b',
            borderRadius: '1rem',
            padding: '2rem',
            border: '1px solid #ef4444'
          }}>
            <h1 style={{ color: '#ef4444', marginTop: 0, marginBottom: '1rem' }}>
              应用出现错误
            </h1>
            <p style={{ color: '#cbd5e1', marginBottom: '1.5rem' }}>
              很抱歉，应用遇到了一个错误。请刷新页面重试。
            </p>
            {this.state.error && (
              <details style={{ marginBottom: '1.5rem' }}>
                <summary style={{ cursor: 'pointer', color: '#94a3b8', marginBottom: '0.5rem' }}>
                  错误详情
                </summary>
                <pre style={{
                  backgroundColor: '#0f172a',
                  padding: '1rem',
                  borderRadius: '0.5rem',
                  overflow: 'auto',
                  fontSize: '0.875rem',
                  color: '#fca5a5',
                  margin: 0
                }}>
                  {this.state.error.toString()}
                  {this.state.errorInfo?.componentStack}
                </pre>
              </details>
            )}
            <button
              onClick={() => window.location.reload()}
              style={{
                backgroundColor: '#3b82f6',
                color: 'white',
                border: 'none',
                padding: '0.75rem 1.5rem',
                borderRadius: '0.5rem',
                cursor: 'pointer',
                fontSize: '1rem',
                fontWeight: '500'
              }}
            >
              刷新页面
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Global error handlers
window.addEventListener('error', (event) => {
  console.error('Global error:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled promise rejection:', event.reason);
  event.preventDefault();
});

// 确保生命周期被注册到 window.moudleQiankunAppLifeCycles
// vite-plugin-qiankun 应该会自动处理，但手动注册确保兼容性
if (typeof window !== 'undefined') {
  // 设置资源基础路径：qiankun 用 /canvas，独立运行用 Vite 的 base（与构建一致，Railway 根路径部署为 ''）
  const baseUrl = typeof import.meta !== 'undefined' && import.meta.env?.BASE_URL
    ? import.meta.env.BASE_URL.replace(/\/$/, '') || ''
    : '';
  (window as any).__ASSET_BASE_PATH__ = (window as any).__POWERED_BY_QIANKUN__ ? '/canvas' : baseUrl;

  if (!window.moudleQiankunAppLifeCycles) {
    (window as any).moudleQiankunAppLifeCycles = {};
  }
  (window as any).moudleQiankunAppLifeCycles['react-canvas'] = {
    bootstrap,
    mount,
    unmount
  };
  console.log('[React] 生命周期已注册到 window.moudleQiankunAppLifeCycles');
}

// 如果不是 qiankun 环境，直接渲染（独立运行）
if (!window.__POWERED_BY_QIANKUN__) {
  // 初始化 LIGHTX2V_TOKEN：如果用户已登录，使用用户的 accessToken
  initLightX2VToken();

  // 检查环境变量（用于调试）
  console.log('[Canvas App] 环境变量检查（独立运行）:', {
    DEEPSEEK_API_KEY: process.env.DEEPSEEK_API_KEY ? `${process.env.DEEPSEEK_API_KEY.substring(0, 10)}...` : '未设置',
    GEMINI_API_KEY: process.env.GEMINI_API_KEY ? `${process.env.GEMINI_API_KEY.substring(0, 10)}...` : '未设置',
    PPCHAT_API_KEY: process.env.PPCHAT_API_KEY ? `${process.env.PPCHAT_API_KEY.substring(0, 10)}...` : '未设置',
    LIGHTX2V_URL: process.env.LIGHTX2V_URL || '未设置',
  });

  const rootElement = document.getElementById('root');
  if (!rootElement) {
    throw new Error("Could not find root element to mount to");
  }

  const standaloneRoot = ReactDOM.createRoot(rootElement);
  standaloneRoot.render(
    <React.StrictMode>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </React.StrictMode>
  );
}
