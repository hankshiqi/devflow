// 学习导读｜Next.js 构建/运行配置；这是框架配置，不属于浏览器页面逻辑。
/** @type {import('next').NextConfig} */
const nextConfig = {
  // Keep the dev server output separate from production build artifacts.
  distDir: process.env.NODE_ENV === "development" ? ".next-dev" : ".next",
  env: {
    NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"
  }
};

module.exports = nextConfig;
