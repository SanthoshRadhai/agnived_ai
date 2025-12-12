"use client";
import React from "react";
import { Satellite, Sun, Moon, Settings, MapPin, Leaf, Video } from "lucide-react";

type HeaderProps = {
  theme: "light" | "dark";
  setTheme: (theme: "light" | "dark") => void;
  activeTab: string;
  setActiveTab: (tab: string) => void;
};

export default function Header({ theme, setTheme, activeTab, setActiveTab }: HeaderProps) {
  return (
    <header className={`border-b ${theme === "dark" ? "border-slate-800 bg-slate-900/80" : "border-gray-200 bg-white/80"} backdrop-blur-sm sticky top-0 z-50`}>
      <div className="flex items-center justify-between px-6 py-3">
        <div className="flex items-center gap-3">
          <MapPin className="w-6 h-6 text-green-400" />
          <span className="font-bold text-lg tracking-tight text-green-200">AgniVed</span>
          <p className="text-xs text-green-300">Geospatial & Wildlife Analysis</p>
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="rounded-full p-2 hover:bg-green-900/30 transition"
            aria-label="Toggle theme"
          >
            {theme === "dark" ? <Sun className="w-5 h-5 text-yellow-300" /> : <Moon className="w-5 h-5 text-slate-700" />}
          </button>
          <nav className="flex gap-2">
            <button
              className={`px-3 py-1 rounded-md text-sm font-medium ${activeTab === "landcover" ? "bg-green-700 text-white" : "bg-transparent text-green-200 hover:bg-green-800/40"}`}
              onClick={() => setActiveTab("landcover")}
            >
              <Satellite className="inline w-4 h-4 mr-1" /> Landcover
            </button>
            <button
              className={`px-3 py-1 rounded-md text-sm font-medium ${activeTab === "vegetation" ? "bg-green-700 text-white" : "bg-transparent text-green-200 hover:bg-green-800/40"}`}
              onClick={() => setActiveTab("vegetation")}
            >
              <Leaf className="inline w-4 h-4 mr-1" /> Vegetation
            </button>
            <button
              className={`px-3 py-1 rounded-md text-sm font-medium ${activeTab === "wildlife" ? "bg-green-700 text-white" : "bg-transparent text-green-200 hover:bg-green-800/40"}`}
              onClick={() => setActiveTab("wildlife")}
            >
              <Video className="inline w-4 h-4 mr-1" /> Wildlife
            </button>
          </nav>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-xs text-green-300">Connected</span>
          </div>
          <button className="rounded-full p-2 hover:bg-green-900/30 transition" aria-label="Settings">
            <Settings className="w-5 h-5 text-green-300" />
          </button>
        </div>
      </div>
    </header>
  );
}