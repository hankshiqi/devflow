// 学习导读｜Next.js 根布局，提供所有页面共享的 HTML 外壳和全局样式入口。
import type { Metadata } from "next";
import "./globals.css";
import { AppShell } from "@/components/layout/app-shell";

export const metadata: Metadata = {
  title: "DevFlow AI",
  description: "PR / Issue intelligent collaboration agent for engineering teams"
};

// Next.js 根布局：设置 HTML/body 外壳并加载全局样式。
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
