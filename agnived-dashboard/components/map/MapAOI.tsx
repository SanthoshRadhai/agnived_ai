// components/map/MapAOI.tsx
'use client';

import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Circle, Marker, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: '/leaflet/marker-icon-2x.png',
  iconUrl: '/leaflet/marker-icon.png',
  shadowUrl: '/leaflet/marker-shadow.png',
});

interface MapAOIProps {
  center: [number, number]; // [lat, lon]
  buffer_km: number;
  onMapClick?: (lat: number, lon: number) => void;
}

function MapController({ center }: { center: [number, number] }) {
  const map = useMap();
  
  useEffect(() => {
    map.setView(center, map.getZoom());
  }, [center, map]);
  
  return null;
}

export default function MapAOI({ center, buffer_km, onMapClick }: MapAOIProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="w-full h-96 bg-slate-800 rounded-lg flex items-center justify-center">
        <p className="text-slate-400">Loading map...</p>
      </div>
    );
  }

  const handleMapClick = (e: L.LeafletMouseEvent) => {
    onMapClick?.(e.latlng.lat, e.latlng.lng);
  };

  return (
    <MapContainer
      center={center}
      zoom={12}
      className="w-full h-96 rounded-lg"
      style={{ height: '400px' }}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      
      <Marker position={center} />
      
      <Circle
        center={center}
        radius={buffer_km * 1000} // Convert km to meters
        pathOptions={{
          color: '#3b82f6',
          fillColor: '#3b82f6',
          fillOpacity: 0.2,
        }}
      />
      
      <MapController center={center} />
    </MapContainer>
  );
}