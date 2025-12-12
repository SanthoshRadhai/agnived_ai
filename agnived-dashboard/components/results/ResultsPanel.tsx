import { useEffect, useState } from "react";
import { api } from "@/../lib/api/backend";
import ResultsPanel from "@/../components/results/ResultsPanel";

function DashboardResultsView({ results, theme }) {
  const [panelData, setPanelData] = useState<{ stats: any; files: string[] } | null>(null);

  useEffect(() => {
    if (!results) return;

    // Fetch metadata for landcover stats
    const fetchStats = async () => {
      let stats = {};
      try {
        const metadata = await api.getMetadata(results.landcover.metadata);
        stats = Object.fromEntries(
          Object.entries(metadata.statistics).map(([k, v]: any) => [
            k,
            { pct: v.percentage, area: v.area_km2 },
          ])
        );
      } catch {
        // fallback: empty stats
      }

      // Collect all relevant files
      const files = [
        results.landcover.visualization,
        results.landcover.sentinel2,
        results.landcover.classification,
        results.landcover.probabilities,
        results.landcover.vegetation_mask,
        results.landcover.metadata,
        results.vegetation.viz_path,
        results.vegetation.cube_path,
      ].filter(Boolean);

      setPanelData({ stats, files });
    };

    fetchStats();
  }, [results]);

  if (!panelData) return null;

  return <ResultsPanel results={panelData} theme={theme} />;
}

export default DashboardResultsView;