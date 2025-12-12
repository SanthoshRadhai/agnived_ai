"use client"

import React, { useState } from "react";
import { AlertCircle, Info } from "lucide-react";

type AOI = {
  lon: number;
  lat: number;
  buffer_km: number;
};

type Layers = {
  vegetation: boolean;
  changeDetection: boolean;
  animalInference: boolean;
};

type SidebarProps = {
  aoi: AOI;
  setAoi: (aoi: AOI) => void;
  layers: Layers;
  setLayers: (layers: Layers) => void;
  isRunning: boolean;
  setIsRunning: (v: boolean) => void;
  logs: { time: string; message: string }[];
  setLogs: (logs: { time: string; message: string }[]) => void;
  setResults: (results: any) => void;
};

export default function Sidebar({
  aoi, setAoi, layers, setLayers, isRunning, setIsRunning, logs, setLogs, setResults,
}: SidebarProps) {
  const [sidebarView, setSidebarView] = useState<"aoi" | "summary">("aoi");
  const toggleLayer = (key: keyof Layers) => setLayers({ ...layers, [key]: !layers[key] });

  return (
    <div className="rounded-md p-4 shadow-xl bg-green-950 border border-green-900">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-white">Sidebar</h2>
        <div className="text-xs text-green-300">30%</div>
      </div>
      <div className="mb-4">
        <label className="text-sm text-green-300 mb-2 block">View</label>
        <select
          value={sidebarView}
          onChange={e => setSidebarView(e.target.value as "aoi" | "summary")}
          className="w-full px-3 py-2 rounded-md border bg-green-900 border-green-800 text-white"
        >
          <option value="aoi">AOI Selector</option>
          <option value="summary">Summary</option>
        </select>
      </div>
      {sidebarView === "aoi" && (
        <>
          <div className="mb-4">
            <h3 className="font-semibold mb-2 text-white">AOI Selector</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-green-300 mb-1">Longitude</label>
                <input
                  type="number"
                  step="0.0001"
                  value={aoi.lon}
                  onChange={e =>
                    setAoi({
                      ...aoi,
                      lon: e.target.value === "" ? 0 : parseFloat(e.target.value),
                    })
                  }
                  className="w-full px-3 py-2 rounded-md border bg-green-900 border-green-800 text-white"
                />
              </div>
              <div>
                <label className="block text-sm text-green-300 mb-1">Latitude</label>
                <input
                  type="number"
                  step="0.0001"
                  value={aoi.lat}
                  onChange={e =>
                    setAoi({
                      ...aoi,
                      lat: e.target.value === "" ? 0 : parseFloat(e.target.value),
                    })
                  }
                  className="w-full px-3 py-2 rounded-md border bg-green-900 border-green-800 text-white"
                />
              </div>
              <div>
                <label className="block text-sm text-green-300 mb-1">Buffer (km)</label>
                <input
                  type="number"
                  step="0.1"
                  min="0.6"
                  value={aoi.buffer_km}
                  onChange={e =>
                    setAoi({
                      ...aoi,
                      buffer_km: e.target.value === "" ? 0 : parseFloat(e.target.value),
                    })
                  }
                  className={`w-full px-3 py-2 rounded-md border ${
                    aoi.buffer_km < 0.6
                      ? "border-red-500 bg-red-500/10"
                      : "bg-green-900 border-green-800 text-white"
                  }`}
                />
              </div>
            </div>
          </div>
          <div className="mb-4">
            <h3 className="font-semibold mb-2 text-white">Layers</h3>
            <div className="space-y-2">
              <label className="flex items-center gap-3">
                <input
                  type="checkbox"
                  checked={layers.vegetation}
                  onChange={() => toggleLayer("vegetation")}
                />
                <span className="text-sm text-white">Vegetation</span>
              </label>
              <label className="flex items-center gap-3">
                <input
                  type="checkbox"
                  checked={layers.changeDetection}
                  onChange={() => toggleLayer("changeDetection")}
                />
                <span className="text-sm text-white">Change Detection</span>
              </label>
              <label className="flex items-center gap-3">
                <input
                  type="checkbox"
                  checked={layers.animalInference}
                  onChange={() => toggleLayer("animalInference")}
                />
                <span className="text-sm text-white">animal Inference</span>
              </label>
            </div>
          </div>
          <div className="mb-4">
            <h3 className="font-semibold mb-2 text-white">Controls</h3>
            <div className="flex flex-col gap-3">
              <button
                disabled={isRunning || aoi.buffer_km < 0.6}
                className="w-full py-2 rounded-md font-medium transition-all bg-gradient-to-r from-green-700 to-emerald-600 text-white disabled:bg-green-900 disabled:text-green-400"
              >
                {isRunning ? "Running Pipeline..." : "Run Pipeline"}
              </button>
              <button
                onClick={() => {
                  setLogs([]);
                  setResults(null);
                }}
                className="w-full py-2 rounded-md border border-green-800 text-white"
              >
                Reset
              </button>
            </div>
            {aoi.buffer_km < 0.6 && (
              <div className="mt-3 p-2 rounded-md bg-orange-500/10 text-orange-400 text-sm flex items-center gap-2">
                <AlertCircle className="w-4 h-4" />
                Buffer must be at least 0.6 km
              </div>
            )}
          </div>
        </>
      )}
      {sidebarView === "summary" && (
        <div className="rounded-md bg-green-900/80 p-4 text-green-100">
          <div className="flex items-center gap-2 mb-2">
            <Info className="w-4 h-4 text-green-400" />
            <span className="font-semibold text-white">Current AOI & Layers</span>
          </div>
          <div className="mb-2">
            <span className="block text-xs text-green-300">Longitude:</span>
            <span className="block font-mono">{aoi.lon}</span>
          </div>
          <div className="mb-2">
            <span className="block text-xs text-green-300">Latitude:</span>
            <span className="block font-mono">{aoi.lat}</span>
          </div>
          <div className="mb-2">
            <span className="block text-xs text-green-300">Buffer (km):</span>
            <span className="block font-mono">{aoi.buffer_km}</span>
          </div>
          <div className="mb-2">
            <span className="block text-xs text-green-300">Layers:</span>
            <span className="block font-mono">
              {Object.entries(layers)
                .filter(([_, v]) => v)
                .map(([k]) => k)
                .join(", ") || "None"}
            </span>
          </div>
          <div className="mt-4 text-xs text-green-400">
            <span>Switch to "AOI Selector" to edit parameters.</span>
          </div>
        </div>
      )}
    </div>
  );
}