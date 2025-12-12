// lib/contexts/PipelineContext.tsx
'use client';

import { createContext, useContext, useState, ReactNode } from 'react';
import type { LandcoverResults, VegetationResults } from '../api/backend';

interface PipelineState {
  landcoverResults: LandcoverResults | null;
  vegetationResults: VegetationResults | null;
  isRunning: boolean;
  logs: Array<{ time: string; message: string }>;
}

interface PipelineContextType extends PipelineState {
  setLandcoverResults: (results: LandcoverResults | null) => void;
  setVegetationResults: (results: VegetationResults | null) => void;
  setIsRunning: (running: boolean) => void;
  addLog: (message: string) => void;
  clearLogs: () => void;
}

const PipelineContext = createContext<PipelineContextType | undefined>(undefined);

export function PipelineProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<PipelineState>({
    landcoverResults: null,
    vegetationResults: null,
    isRunning: false,
    logs: [],
  });

  const setLandcoverResults = (results: LandcoverResults | null) =>
    setState(prev => ({ ...prev, landcoverResults: results }));

  const setVegetationResults = (results: VegetationResults | null) =>
    setState(prev => ({ ...prev, vegetationResults: results }));

  const setIsRunning = (running: boolean) =>
    setState(prev => ({ ...prev, isRunning: running }));

  const addLog = (message: string) =>
    setState(prev => ({
      ...prev,
      logs: [...prev.logs, { time: new Date().toLocaleTimeString(), message }],
    }));

  const clearLogs = () =>
    setState(prev => ({ ...prev, logs: [] }));

  return (
    <PipelineContext.Provider
      value={{
        ...state,
        setLandcoverResults,
        setVegetationResults,
        setIsRunning,
        addLog,
        clearLogs,
      }}
    >
      {children}
    </PipelineContext.Provider>
  );
}

export function usePipeline() {
  const context = useContext(PipelineContext);
  if (!context) {
    throw new Error('usePipeline must be used within PipelineProvider');
  }
  return context;
}