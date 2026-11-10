// 学习导读｜Tailwind 扫描目录与主题扩展，决定源码中的工具类如何生成 CSS。
import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#17212b",
        panel: "#f7f9fb",
        line: "#d8e0e8",
        accent: "#0f766e",
        warn: "#b45309"
      }
    }
  },
  plugins: []
};

export default config;
