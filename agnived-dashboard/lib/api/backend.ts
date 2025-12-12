import React from "react";

// Set API base URL from env or default to localhost
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// --------------------
// Type Definitions
// --------------------

export interface AOIConfig {
  lon: number;
  lat: number;
  buffer_km?: number;
  buffer_m?: number;
}

export interface LandcoverRequest {
  lon: number;
  lat: number;
  buffer_km: number;
  date_start: string;
  date_end: string;
  scale: number;
  cloud_cover_max: number;
}

export interface VegetationRequest {
  lon: number;
  lat: number;
  buffer_km: number;
  use_mask: boolean;
}

export interface PipelineRequest extends LandcoverRequest {
  veg_buffer_km?: number;
}

export interface VideoRequest {
  youtube_url: string;
}

export interface LandcoverResults {
  status: string;
  outputs: {
    sentinel2: string;
    classification: string;
    probabilities: string;
    vegetation_mask: string;
    visualization: string;
    metadata: string;
  };
  aoi?: AOIConfig;
}

export interface VegetationResults {
  status: string;
  aoi: AOIConfig;
  cube_path: string;
  viz_path: string;
  class_distribution: Record<string, number>;
  tile_counts: Record<string, number>;
  avg_confidence: number;
  tiles_shape: [number, number];
}

export interface PipelineResults {
  status: string;
  aoi: AOIConfig;
  landcover: LandcoverResults["outputs"];
  vegetation: Omit<VegetationResults, "status" | "aoi">;
}

export interface VideoResults {
  status: string;
  youtube_url: string;
  note: string;
}

export interface Metadata {
  aoi: {
    lon: number;
    lat: number;
    buffer_km: number;
    bounds: number[];
  };
  date_range: {
    start: string;
    end: string;
  };
  scale: number;
  statistics: Record<
    string,
    {
      pixels: number;
      area_km2: number;
      percentage: number;
      description: string;
    }
  >;
}

// --------------------
// API Client Class
// --------------------

export class AgniVedAPI {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  // Landcover pipeline
  async runLandcover(config: LandcoverRequest): Promise<LandcoverResults> {
    const response = await fetch(`${this.baseURL}/landcover/dw`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    if (!response.ok) throw new Error(`Landcover pipeline failed: ${response.statusText}`);
    return response.json();
  }

  // Vegetation pipeline
  async runVegetation(config: VegetationRequest): Promise<VegetationResults> {
    const response = await fetch(`${this.baseURL}/vegetation/bigearth`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    if (!response.ok) throw new Error(`Vegetation pipeline failed: ${response.statusText}`);
    return response.json();
  }

  // Full pipeline
  async runPipeline(config: PipelineRequest): Promise<PipelineResults> {
    const response = await fetch(`${this.baseURL}/pipeline/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    if (!response.ok) throw new Error(`Full pipeline failed: ${response.statusText}`);
    return response.json();
  }

  // Start YouTube wildlife detection
  async startVideoClassification(url: string): Promise<VideoResults> {
    const response = await fetch(`${this.baseURL}/video/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ youtube_url: url }),
    });
    if (!response.ok) throw new Error(`Video classification failed: ${response.statusText}`);
    return response.json();
  }

  // Get file (image/GeoTIFF) from backend
  async getFile(path: string): Promise<Blob> {
    const response = await fetch(`${this.baseURL}/files/image?path=${encodeURIComponent(path)}`);
    if (!response.ok) throw new Error(`Failed to fetch file: ${response.statusText}`);
    return response.blob();
  }

  // Get file URL for direct linking
  getFileURL(path: string): string {
    return `${this.baseURL}/files/image?path=${encodeURIComponent(path)}`;
  }

  // Get metadata JSON
  async getMetadata(path: string = "Final_Res_DW/metadata.json"): Promise<Metadata> {
    const blob = await this.getFile(path);
    const text = await blob.text();
    return JSON.parse(text);
  }

  // Download file to user's computer
  async downloadFile(path: string, filename?: string): Promise<void> {
    const blob = await this.getFile(path);
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename || path.split("/").pop() || "download";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  }

  // Check backend health/status
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/`);
      return response.ok;
    } catch {
      return false;
    }
  }
}

// Singleton instance
export const api = new AgniVedAPI();

// React Hook for API calls with loading/error states
export function useAgniVedAPI() {
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const execute = async <T,>(
    apiCall: () => Promise<T>,
    onSuccess?: (data: T) => void
  ): Promise<T | null> => {
    setLoading(true);
    setError(null);

    try {
      const result = await apiCall();
      onSuccess?.(result);
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      console.error("API Error:", err);
      return null;
    } finally {
      setLoading(false);
    }
  };

  return { execute, loading, error };
}

// Utility: Generate cache key for AOI+config
export function generateCacheKey(config: Partial<LandcoverRequest | VegetationRequest>): string {
  const normalized = {
    lon: config.lon?.toFixed(4),
    lat: config.lat?.toFixed(4),
    buffer: config.buffer_km,
    dates: "date_start" in config ? `${config.date_start}_${config.date_end}` : null,
  };
  return Object.values(normalized).filter(Boolean).join("_");
}

// Utility: Parse backend error messages
export function parseAPIError(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "object" && error !== null && "message" in error) return String((error as any).message);
  return "An unknown error occurred";
}

export default api;