import React from "react";
import dynamic from "next/dynamic";

// Dynamically import MapAOI to avoid SSR issues
const MapAOI = dynamic(() => import("./MapAOI"), { ssr: false });

interface MapPanelProps {
  aoi: { lat: number; lon: number; buffer_km: number };
  layers: { vegetation: boolean; changeDetection: boolean; animalInference: boolean };
  setLayers: React.Dispatch<React.SetStateAction<{ vegetation: boolean; changeDetection: boolean; animalInference: boolean }>>;
theme: string;
}

export default function MapPanel({ aoi, layers, setLayers, theme }: MapPanelProps) {
  return (
    <div className={`rounded-lg overflow-hidden shadow-xl ${theme === "dark" ? "bg-slate-900 border border-slate-800" : "bg-white border border-gray-200"}`}>
      <div className="p-4 border-b flex items-center justify-between">
        <h2 className="font-semibold flex items-center gap-2 text-white">
          Map
        </h2>
        <div className="flex items-center gap-3">
          <span className="text-sm text-green-300">
            AOI area: {(Math.PI * Math.pow(aoi.buffer_km, 2)).toFixed(2)} km²
          </span>
        </div>
      </div>
      <div className={`relative ${theme === "dark" ? "bg-slate-800" : "bg-gray-100"}`}>
        <MapAOI center={[aoi.lat, aoi.lon]} buffer_km={aoi.buffer_km}/>
        <div className="absolute right-6 top-6 w-48 p-3 rounded-md border bg-white/90 dark:bg-slate-900/90 shadow-lg">
          <button onClick={() => setLayers(l => ({ ...l, vegetation: !l.vegetation }))}
            className={`w-full py-2 rounded-md mb-2 text-sm font-medium ${layers.vegetation ? "bg-green-600 text-white" : "bg-gray-100"}`}>
            Vegetation
          </button>
          <button onClick={() => setLayers(l => ({ ...l, changeDetection: !l.changeDetection }))}
            className={`w-full py-2 rounded-md mb-2 text-sm font-medium ${layers.changeDetection ? "bg-yellow-500 text-white" : "bg-gray-100"}`}>
            Change Detection
          </button>
          <button onClick={() => setLayers(l => ({ ...l, animalInference: !l.animalInference }))}
            className={`w-full py-2 rounded-md text-sm font-medium ${layers.animalInference ? "bg-blue-600 text-white" : "bg-gray-100"}`}>
            animal Inference
          </button>
        </div>
      </div>
    </div>
  );
}