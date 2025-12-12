"use client";
import React, { useState } from "react";
import Header from "@/../components/layout/Header";
import Sidebar from "@/../components/layout/Sidebar";
import MapPanel from "@/../components/map/MapPanel";
import PipelineStatus from "@/../components/shared/PipelineStatus";
import ResultsPanel from "@/../components/results/ResultsPanel";
import { api, useAgniVedAPI } from "@/../lib/api/backend";

const backgroundStyle = {
  background: "linear-gradient(135deg, #0f2d25 0%, #183c2b 100%)",
  backgroundImage: "url('/bg.png')",
  backgroundRepeat: "no-repeat",
  backgroundSize: "cover",
  minHeight: "100vh",
};

export default function AgniVedDashboard() {
  const [theme, setTheme] = useState("dark");
  const [activeTab, setActiveTab] = useState("landcover");
  const [aoi, setAoi] = useState({ lon: 77.5946, lat: 12.9716, buffer_km: 1.2 });
  const [logs, setLogs] = useState<{ time: string; message: string }[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<any | null>(null);
  const [layers, setLayers] = useState({
    vegetation: true,
    changeDetection: false,
    animalInference: false,
  });

  const { execute, loading, error } = useAgniVedAPI();

  const runPipeline = async () => {
    setIsRunning(true);
    setLogs((prev) => [
      ...prev,
      { time: new Date().toLocaleTimeString(), message: "Running pipeline..." },
    ]);
    setResults(null);

    const config = {
      lon: aoi.lon,
      lat: aoi.lat,
      buffer_km: aoi.buffer_km,
      veg_buffer_km: aoi.buffer_km,
      date_start: "2024-01-01",
      date_end: "2024-01-31",
      scale: 10,
      cloud_cover_max: 20,
    };

    await execute(() => api.runPipeline(config), (data) => {
      setResults(data);
      setLogs((prev) => [
        ...prev,
        { time: new Date().toLocaleTimeString(), message: "Pipeline complete." },
      ]);
      setIsRunning(false);
    });

    setIsRunning(false);
  };

  return (
    <div style={backgroundStyle}>
      <Header theme={theme} setTheme={setTheme} activeTab={activeTab} setActiveTab={setActiveTab} />
      <div className="container mx-auto p-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <aside className="lg:col-span-4 space-y-6">
            <Sidebar
              aoi={aoi}
              setAoi={setAoi}
              layers={layers}
              setLayers={setLayers}
              isRunning={isRunning}
              setIsRunning={setIsRunning}
              logs={logs}
              setLogs={setLogs}
              setResults={setResults}
            />
            <PipelineStatus logs={logs} setLogs={setLogs} />
            <button
              onClick={runPipeline}
              disabled={isRunning || loading}
              className="w-full py-2 rounded-md font-medium transition-all bg-gradient-to-r from-green-700 to-emerald-600 text-white disabled:bg-green-900 disabled:text-green-400"
            >
              {isRunning || loading ? "Running..." : "Run Pipeline"}
            </button>
            {error && <div className="text-red-500 mt-2">{error}</div>}
          </aside>
          <main className="lg:col-span-8 space-y-6">
            <MapPanel aoi={aoi} layers={layers} setLayers={setLayers} theme={theme} />
            {(isRunning || loading) && (
              <div className="flex items-center justify-center h-96">
                <span className="text-green-300 text-lg animate-pulse">Loading results...</span>
              </div>
            )}
            {results && (
              <div className="space-y-6">
                {/* Show landcover visualization */}
                {results.landcover?.visualization && (
                  <img
                    src={api.getFileURL(results.landcover.visualization)}
                    alt="Landcover Visualization"
                    className="w-full rounded-lg shadow-lg"
                  />
                )}
                {/* Show vegetation visualization */}
                {results.vegetation?.viz_path && (
                  <img
                    src={api.getFileURL(results.vegetation.viz_path)}
                    alt="Vegetation Classification"
                    className="w-full rounded-lg shadow-lg"
                  />
                )}
                {/* Optionally, show results panel */}
                <ResultsPanel results={results} theme={theme} />
              </div>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}