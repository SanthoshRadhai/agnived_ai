import { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Circle, Marker, ImageOverlay, useMapEvents } from 'react-leaflet';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

import '../styles/Dashboard.css';

// --- ASSETS ---
import bengalTigerImg from '../assets/bengal-tiger.png';
import indianElephantImg from '../assets/indian-elephant.png';
import asiaticLionImg from '../assets/asiatic-lion.png';
import indianRhinoImg from '../assets/indian-rhino.png';
import snowLeopardImg from '../assets/snow-leopard.png';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

// --- LEAFLET ICON FIX ---
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

// --- DATA: INFORMATIC CARDS ---
const SPECIES_DATA = [
  { 
    name: "Bengal Tiger", 
    status: "Endangered",
    habitat: "Mangroves & Grasslands",
    note: "Keystone species crucial for controlling herbivore populations.",
    img: bengalTigerImg
  },
  { 
    name: "Indian Elephant", 
    status: "Endangered",
    habitat: "Forests",
    note: "Engineers of the forest, creating pathways for other animals.",
    img: indianElephantImg
  },
  { 
    name: "Asiatic Lion", 
    status: "Endangered",
    habitat: "Dry Deciduous Forests",
    note: "Found only in Gir National Park, living symbol of pride.",
    img: asiaticLionImg
  },
  { 
    name: "Indian Rhino", 
    status: "Vulnerable",
    habitat: "Grasslands",
    note: "The largest of the rhino species, identified by a single horn.",
    img: indianRhinoImg
  },
  { 
    name: "Snow Leopard", 
    status: "Vulnerable",
    habitat: "High Himalayas",
    note: "Known as the 'Ghost of the Mountains' due to elusive nature.",
    img: snowLeopardImg
  },
];

// Helper: Handle Map Clicks
function LocationSelector({ onLocationSelect }) {
  useMapEvents({
    click(e) {
      onLocationSelect(e.latlng);
    },
  });
  return null;
}

// Helper: Secure Image Component (Fetches image with Bearer Token)
const SecureImage = ({ imageId, alt }) => {
  const [imgSrc, setImgSrc] = useState(null);

  useEffect(() => {
    const fetchImage = async () => {
      try {
        const token = localStorage.getItem('token');
        const response = await axios.get(`${API_URL}/image/${imageId}`, {
          headers: { Authorization: `Bearer ${token}` },
          responseType: 'blob'
        });
        const url = URL.createObjectURL(response.data);
        setImgSrc(url);
      } catch (err) {
        console.error("Failed to load image", err);
      }
    };
    if (imageId) fetchImage();
  }, [imageId]);

  if (!imgSrc) return <div className="img-placeholder">Loading...</div>;
  return <img src={imgSrc} alt={alt} />;
};

// --- HELPER: CONVERT BACKEND PATH TO URL ---
const getFilename = (fullPath) => {
  if (!fullPath) return "";
  // Removes "D:\Projects\..." and leaves just "image.png"
  return fullPath.split(/[/\\]/).pop();
};

export default function Dashboard() {
  // --- STATE: GLOBAL ---
  const [activeTab, setActiveTab] = useState('analysis');
  const [viewState, setViewState] = useState('map'); 
  
  // --- STATE: GEOSPATIAL ---
  const [coords, setCoords] = useState({ lat: 28.56027870, lng: 77.29239823 });
  const [radius, setRadius] = useState(3);
  const [showLabels, setShowLabels] = useState(false);
  const [mapOverlays, setMapOverlays] = useState({
    landcover: null,  // Base64 string
    vegetation: null, // Base64 string
    bounds: null      // [ [lat, lon], [lat, lon] ]
  });

  // --- STATE: UPLOAD ---
  const [uploadForm, setUploadForm] = useState({ species: '', file: null });
  const [myUploads, setMyUploads] = useState([]);
  const [showGalleryModal, setShowGalleryModal] = useState(false);

  // --- STATE: MODULE CONFIGURATION ---
  const [modules, setModules] = useState({
    landcover: true,
    vegetation: false,
    uploads: true,
    panos: false,
    changeDetection: false,
    wildlifeStream: false
  });

  const [config, setConfig] = useState({
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    streamUrl: '',
    // Panos configuration
    panos_count: 3,
    panos_area_of_interest: 100.0,
    panos_min_distance: 20.0,
    panos_labels: {
      tree: true,
      bushes: true,
      vegetation: true
    }
  });

  // --- STATE: EXECUTION & RESULTS ---
  const [loading, setLoading] = useState({
    landcover: false, vegetation: false, uploads: false, panos: false, changeDetection: false, wildlifeStream: false
  });

  const [results, setResults] = useState({
    landcover: null, vegetation: null, uploads: [], panos: null, changeDetection: null, wildlifeStream: null
  });

  const [expandedWidget, setExpandedWidget] = useState(null);
  
  // --- STATE: PANOS MODAL ---
  const [showPanosModal, setShowPanosModal] = useState(false);
  const [panosModalTab, setPanosModalTab] = useState('panoramas'); // 'panoramas' or 'plants'

  // --- EFFECTS ---
  useEffect(() => {
    if (activeTab === 'upload') {
      fetchMyUploads();
    }
  }, [activeTab]);

  const fetchMyUploads = async () => {
    try {
      const token = localStorage.getItem('token');
      const res = await axios.get(`${API_URL}/uploads/me?limit=10`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setMyUploads(res.data.uploads || []);
    } catch (err) {
      console.error("Error fetching uploads", err);
    }
  };

  // --- HANDLERS: CONFIG ---
  const toggleModule = (key) => {
    setModules(prev => {
      const newState = { ...prev, [key]: !prev[key] };
      // Logic: If Vegetation is True, Landcover MUST be True
      if (key === 'vegetation' && newState.vegetation) {
        newState.landcover = true;
      }
      // Logic: If Landcover is unchecked, Vegetation MUST be unchecked
      if (key === 'landcover' && !newState.landcover) {
        newState.vegetation = false;
      }
      return newState;
    });
  };

  const togglePanosLabel = (label) => {
    setConfig(prev => ({
      ...prev,
      panos_labels: {
        ...prev.panos_labels,
        [label]: !prev.panos_labels[label]
      }
    }));
  };

  const handleMapClick = (latlng) => {
    setCoords({ lat: latlng.lat, lng: latlng.lng });
  };

  // --- HANDLERS: EXECUTION ---
  const handleRunAnalysis = async () => {
    setViewState('grid'); // Switch to Grid View
    const token = localStorage.getItem('token');
    const headers = { Authorization: `Bearer ${token}` };

    // Reset Results & Set Loading based on selected modules
    const newLoading = {};
    Object.keys(modules).forEach(key => {
      if (modules[key]) newLoading[key] = true;
    });
    setLoading(newLoading);

    // 1. LANDCOVER
    if (modules.landcover) {
      axios.post(`${API_URL}/run_landcover`, {
        lon: coords.lng, lat: coords.lat, buffer_km: radius,
        date_start: config.startDate, date_end: config.endDate
      }, { headers }).then(res => {
        setResults(prev => ({ ...prev, landcover: res.data }));
        if (res.data.image_base64) {
            setMapOverlays(prev => ({
                ...prev,
                landcover: `data:image/png;base64,${res.data.image_base64}`,
                // Ideally backend should return bounds, for now using rough estimate based on radius
                // NOTE: This assumes the image is centered on the coords. Adjust logic if needed.
                bounds: [[coords.lat - (radius/111), coords.lng - (radius/111)], [coords.lat + (radius/111), coords.lng + (radius/111)]]
            }));
        }
      }).catch(err => console.error(err))
      .finally(() => setLoading(prev => ({ ...prev, landcover: false })));
    }

    // 2. VEGETATION
    if (modules.vegetation) {
      axios.post(`${API_URL}/run_vegetation`, {
        lon: coords.lng, lat: coords.lat, buffer_km: radius
      }, { headers }).then(res => {
        setResults(prev => ({ ...prev, vegetation: res.data }));
        if (res.data.image_base64) {
            setMapOverlays(prev => ({
                ...prev,
                vegetation: `data:image/png;base64,${res.data.image_base64}`,
                bounds: [[coords.lat - (radius/111), coords.lng - (radius/111)], [coords.lat + (radius/111), coords.lng + (radius/111)]]
            }));
        }
      }).catch(err => console.error(err))
      .finally(() => setLoading(prev => ({ ...prev, vegetation: false })));
    }

    // 3. UPLOADS
    if (modules.uploads) {
      axios.post(`${API_URL}/uploads/search`, {
        latitude: coords.lat, longitude: coords.lng, radius_km: radius
      }, { headers }).then(res => {
        setResults(prev => ({ ...prev, uploads: res.data.uploads || [] }));
      }).catch(err => console.error(err))
      .finally(() => setLoading(prev => ({ ...prev, uploads: false })));
    }

    // 4. PANOS (360)
    if (modules.panos) {
      const selectedLabels = Object.keys(config.panos_labels).filter(label => config.panos_labels[label]);
      axios.post(`${API_URL}/run_panos_and_plant_identification`, {
        lat: coords.lat,
        lon: coords.lng,
        panos_lat: coords.lat,
        panos_lon: coords.lng,
        panos_count: config.panos_count,
        panos_area_of_interest: config.panos_area_of_interest,
        panos_min_distance: config.panos_min_distance,
        panos_labels: selectedLabels,
        buffer_km: radius,
        date_start: config.startDate,
        date_end: config.endDate
      }, { headers }).then(res => {
        setResults(prev => ({ ...prev, panos: res.data }));
      }).catch(err => console.error(err))
      .finally(() => setLoading(prev => ({ ...prev, panos: false })));
    }

    // 5. CHANGE DETECTION
    if (modules.changeDetection) {
      // Simulating heavy process
      setTimeout(() => {
        setResults(prev => ({ 
          ...prev, 
          changeDetection: { status: "Detected", percent_change: 12.5, map_image: "placeholder_diff.png" } 
        }));
        setLoading(prev => ({ ...prev, changeDetection: false }));
      }, 3500);
    }

    // 6. WILDLIFE STREAM
    if (modules.wildlifeStream) {
       setTimeout(() => {
        setResults(prev => ({ ...prev, wildlifeStream: config.streamUrl }));
        setLoading(prev => ({ ...prev, wildlifeStream: false }));
       }, 1000);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!uploadForm.file) return alert("Please select a file");

    try {
      const token = localStorage.getItem('token');
      const formData = new FormData();
      formData.append('image', uploadForm.file);
      formData.append('latitude', coords.lat);
      formData.append('longitude', coords.lng);
      if(uploadForm.species) formData.append('species', uploadForm.species);

      await axios.post(`${API_URL}/upload`, formData, {
        headers: { 
          Authorization: `Bearer ${token}`,
          'Content-Type': 'multipart/form-data'
        }
      });

      alert("Upload Successful!");
      setUploadForm({ species: '', file: null });
      fetchMyUploads();
    } catch (err) {
      alert("Upload failed");
    }
  };

  // --- RENDERERS ---

  const renderInformaticCarousel = () => (
    <div className="info-carousel-container">
      <div className="info-track">
        {[...SPECIES_DATA, ...SPECIES_DATA].map((item, idx) => (
          <div key={idx} className="species-info-card">
            <div className="card-image">
              <img src={item.img} alt={item.name} />
              <span className={`status-tag ${item.status.toLowerCase()}`}>{item.status}</span>
            </div>
            <div className="card-details">
              <h5>{item.name}</h5>
              <small>📍 {item.habitat}</small>
              <p>{item.note}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderWidgetContent = (type) => {
    if (loading[type]) return <div className="widget-loader"><div className="spinner"></div><p>Processing Pipeline...</p></div>;
    if (!modules[type]) return <div className="widget-inactive"><p>Module Inactive</p></div>;

    const data = results[type];
    if (!data && type !== 'wildlifeStream') return <div className="widget-waiting"><p>Waiting for data...</p></div>;

    switch(type) {
      case 'landcover':
        return (
          <div className="widget-content">
            <h4>Landcover Classification</h4>
            {/* UPDATED: Use base64 string */}
            {data?.image_base64 ? (
               <img 
                 src={`data:image/png;base64,${data.image_base64}`} 
                 alt="Landcover Map" 
                 className="result-img" 
               />
            ) : <p className="data-text">Map data unavailable.</p>}
            <div className="stat-overlay">Cloud Cover: {data?.cloud_cover_max || 0}%</div>
          </div>
        );
      case 'vegetation':
        return (
          <div className="widget-content">
            <h4>Vegetation Index</h4>
            {/* UPDATED: Use base64 string */}
            {data?.image_base64 ? (
               <img 
                 src={`data:image/png;base64,${data.image_base64}`} 
                 alt="Vegetation Map" 
                 className="result-img" 
               />
            ) : <p className="data-text">Index unavailable.</p>}
            <div className="stat-overlay">Confidence: {((data?.avg_confidence || 0) * 100).toFixed(1)}%</div>
          </div>
        );
      case 'uploads':
        return (
          <div className="widget-content">
            <h4>Community Sightings</h4>
            <div className="collage-container">
              {data.length > 0 ? (
                <div className="collage-grid">
                  {/* Show max 9 images in preview, or all if expanded */}
                  {data.slice(0, expandedWidget === 'uploads' ? 50 : 9).map((img, i) => (
                    <div key={i} className="collage-item">
                      <SecureImage imageId={img.id} alt={img.species} />
                      <div className="species-hover-overlay">
                        <span className="species-name">{img.species || 'Unknown Species'}</span>
                        <span className="upload-date">{new Date(img.created_at).toLocaleDateString()}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="no-data-placeholder">
                  <span style={{fontSize: '2rem'}}>🔭</span>
                  <p>No community sightings in this radius ({radius}km).</p>
                </div>
              )}
            </div>
            {/* Show count overlay if not expanded */}
            {data.length > 9 && expandedWidget !== 'uploads' && (
              <div className="more-count">+{data.length - 9} more</div>
            )}
          </div>
        );
        
      case 'panos':
        return (
          <div className="widget-content panos-widget">
            <h4>360° Ground Truth</h4>
            {loading.panos ? (
              <div className="panos-loading">
                <div className="spinner-large"></div>
                <p>Scanning panoramas & identifying plants...</p>
              </div>
            ) : data?.panos ? (
              <div className="panos-results-summary">
                <div className="stat-box">
                  <span className="stat-number">{data.panos.pano_result?.count_returned || 0}</span>
                  <span className="stat-label">Panoramas Found</span>
                </div>
                <div className="stat-box">
                  <span className="stat-number">{data.summary?.total_objects_detected || 0}</span>
                  <span className="stat-label">Objects Detected</span>
                </div>
                <div className="stat-box">
                  <span className="stat-number">{data.summary?.total_plants_identified || 0}</span>
                  <span className="stat-label">Plants Identified</span>
                </div>
                <button className="view-results-btn" onClick={() => setShowPanosModal(true)}>
                  View Results 🔍
                </button>
              </div>
            ) : (
              <p className="data-text">No panorama data available.</p>
            )}
          </div>
        );
      case 'changeDetection':
        return (
          <div className="widget-content">
            <h4>Change Detection</h4>
            <div className="stat-big danger">{data.percent_change}%</div>
            <p className="data-text">Forest Loss Detected</p>
            <p className="date-range">{config.startDate} to {config.endDate}</p>
          </div>
        );
      case 'wildlifeStream':
        return (
          <div className="widget-content video-mode">
            <h4>Live Feed</h4>
            {data ? (
               <iframe width="100%" height="100%" src={`https://www.youtube.com/embed/${data.split('v=')[1] || ''}?autoplay=1&mute=1`} title="Wildlife" frameBorder="0" allow="autoplay; encrypted-media"></iframe>
            ) : <p>Invalid Stream URL</p>}
          </div>
        );
      default: return null;
    }
  };

  return (
    <div className="dashboard-container">
      {/* --- PANOS MODAL --- */}
      {showPanosModal && results.panos?.panos && (
        <div className="modal-overlay panos-modal-overlay" onClick={() => setShowPanosModal(false)}>
          <div className="modal-content panos-modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header panos-modal-header">
              <div className="header-top">
                <h3>360° Ground Truth Analysis</h3>
                <button className="modal-close-btn" onClick={() => setShowPanosModal(false)}>×</button>
              </div>
              <div className="modal-tabs">
                <button 
                  className={`tab-btn ${panosModalTab === 'panoramas' ? 'active' : ''}`}
                  onClick={() => setPanosModalTab('panoramas')}
                >
                  📸 Panoramas ({results.panos.panos.pano_result?.count_returned || 0})
                </button>
                <button 
                  className={`tab-btn ${panosModalTab === 'plants' ? 'active' : ''}`}
                  onClick={() => setPanosModalTab('plants')}
                >
                  🌱 Identified Plants ({results.panos.summary?.total_plants_identified || 0})
                </button>
              </div>
            </div>

            <div className="modal-body panos-modal-body">
              {panosModalTab === 'panoramas' && (
                <div className="panoramas-section">
                  <h4>Panoramic Images</h4>
                  <div className="panoramas-grid">
                    {results.panos.panos.detected_objects?.map((obj, idx) => (
                      <div key={idx} className="panorama-card">
                        {obj.pano_image_base64 ? (
                          <img 
                            src={`data:image/jpeg;base64,${obj.pano_image_base64}`} 
                            alt={`Panorama ${obj.pano_id}`}
                            className="panorama-image"
                          />
                        ) : (
                          <div className="image-placeholder">No Image</div>
                        )}
                        <div className="panorama-info">
                          <small className="pano-id">ID: {obj.pano_id?.slice(0, 8)}...</small>
                          <small className="object-count">🎯 {obj.object_detection?.num_crops || 0} Objects</small>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {panosModalTab === 'plants' && (
                <div className="plants-section">
                  <h4>Identified Plant Species</h4>
                  <div className="plants-list">
                    {results.panos.panos.plant_identification_results?.map((pano_result, pano_idx) => (
                      <div key={pano_idx} className="pano-plant-group">
                        <div className="pano-group-header">
                          <h5>📍 Panorama {pano_idx + 1}</h5>
                          <span className="crop-count">{pano_result.crop_plant_identifications?.length || 0} crops</span>
                        </div>
                        <div className="crops-grid">
                          {pano_result.crop_plant_identifications?.map((crop, crop_idx) => (
                            <div key={crop_idx} className="plant-card">
                              {crop.crop_image_base64 ? (
                                <img 
                                  src={`data:image/jpeg;base64,${crop.crop_image_base64}`} 
                                  alt={`Crop ${crop_idx}`}
                                  className="plant-crop-image"
                                />
                              ) : (
                                <div className="image-placeholder">No Crop</div>
                              )}
                              <div className="plant-details">
                                <div className="crop-label-badge">{crop.object_label || 'Unknown'}</div>
                                {crop.top_prediction ? (
                                  <div className="prediction-info">
                                    <div className="species-name">{crop.top_prediction.species}</div>
                                    <div className="confidence-text">{crop.top_prediction.confidence_percentage}</div>
                                  </div>
                                ) : crop.error ? (
                                  <div className="error-text">⚠️ Error</div>
                                ) : null}
                                
                                {crop.predictions && crop.predictions.length > 1 && (
                                  <div className="other-predictions">
                                    <small className="pred-title">Other Matches:</small>
                                    {crop.predictions.slice(1, 3).map((pred, idx) => (
                                      <div key={idx} className="pred-item">
                                        <small>{pred.species}</small>
                                        <small className="pred-confidence">{pred.confidence_percentage}</small>
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="modal-footer panos-modal-footer">
              <p className="summary-text">
                Total: {results.panos.summary?.total_panos || 0} panoramas | 
                {results.panos.summary?.total_objects_detected || 0} objects | 
                {results.panos.summary?.total_plants_identified || 0} identifications
              </p>
              <button className="close-modal-btn" onClick={() => setShowPanosModal(false)}>Close</button>
            </div>
          </div>
        </div>
      )}
      <div className="control-panel">
        <div className="brand"><h2>🌿 AgniVed Dashboard</h2></div>
        <div className="tabs">
          <button className={activeTab === 'analysis' ? 'active' : ''} onClick={() => setActiveTab('analysis')}>Analysis</button>
          <button className={activeTab === 'upload' ? 'active' : ''} onClick={() => setActiveTab('upload')}>Upload</button>
        </div>

        <div className="panel-content">
          {activeTab === 'analysis' ? (
            <div className="analysis-config">
              <h3>Run Geospatial Analysis</h3>
              <div className="input-row">
                <div className="group"><label>Lat</label><input type="number" value={coords.lat} onChange={(e) => setCoords({...coords, lat: parseFloat(e.target.value)})} /></div>
                <div className="group"><label>Lon</label><input type="number" value={coords.lng} onChange={(e) => setCoords({...coords, lng: parseFloat(e.target.value)})} /></div>
              </div>
              <div className="group range-group">
                <label>Radius: {radius} km</label>
                <input type="range" min="3" max="20" value={radius} onChange={(e) => setRadius(parseInt(e.target.value))} />
              </div>

              <div className="module-selector">
                <div className="config-group">
                  <h5>Satellite Intelligence</h5>
                  <label className="checkbox-item"><input type="checkbox" checked={modules.landcover} onChange={() => toggleModule('landcover')} /><span>S2 Landcover Classification</span></label>
                  <label className="checkbox-item"><input type="checkbox" checked={modules.vegetation} onChange={() => toggleModule('vegetation')} /><span>S2 Vegetation (Requires Landcover)</span></label>
                </div>
                <div className="config-group">
                  <h5>Ground Truth</h5>
                  <label className="checkbox-item"><input type="checkbox" checked={modules.uploads} onChange={() => toggleModule('uploads')} /><span>Community Sightings</span></label>
                  <label className="checkbox-item"><input type="checkbox" checked={modules.panos} onChange={() => toggleModule('panos')} /><span>360° Ground Truth</span></label>
                  
                  {/* PANOS CONFIGURATION */}
                  {modules.panos && (
                    <div className="sub-config panos-config">
                      <div className="config-input-group">
                        <label>Panorama Count</label>
                        <input 
                          type="number" 
                          min="1" 
                          max="10" 
                          value={config.panos_count} 
                          onChange={(e) => setConfig({...config, panos_count: parseInt(e.target.value)})} 
                        />
                      </div>
                      
                      <div className="config-input-group">
                        <label>Area of Interest (m)</label>
                        <input 
                          type="number" 
                          min="10" 
                          max="500" 
                          value={config.panos_area_of_interest} 
                          onChange={(e) => setConfig({...config, panos_area_of_interest: parseFloat(e.target.value)})} 
                        />
                      </div>
                      
                      <div className="config-input-group">
                        <label>Min Distance (m)</label>
                        <input 
                          type="number" 
                          min="5" 
                          max="100" 
                          value={config.panos_min_distance} 
                          onChange={(e) => setConfig({...config, panos_min_distance: parseFloat(e.target.value)})} 
                        />
                      </div>
                      
                      <div className="labels-config">
                        <label className="section-label">Detection Labels</label>
                        <label className="checkbox-item">
                          <input 
                            type="checkbox" 
                            checked={config.panos_labels.tree} 
                            onChange={() => togglePanosLabel('tree')} 
                          />
                          <span>🌳 Tree</span>
                        </label>
                        <label className="checkbox-item">
                          <input 
                            type="checkbox" 
                            checked={config.panos_labels.bushes} 
                            onChange={() => togglePanosLabel('bushes')} 
                          />
                          <span>🌿 Bushes</span>
                        </label>
                        <label className="checkbox-item">
                          <input 
                            type="checkbox" 
                            checked={config.panos_labels.vegetation} 
                            onChange={() => togglePanosLabel('vegetation')} 
                          />
                          <span>🍃 Vegetation</span>
                        </label>
                      </div>
                    </div>
                  )}
                </div>
                <div className="config-group">
                  <h5>Temporal & Live</h5>
                  <label className="checkbox-item"><input type="checkbox" checked={modules.changeDetection} onChange={() => toggleModule('changeDetection')} /><span>Change Detection Pipeline</span></label>
                  {modules.changeDetection && (<div className="sub-config"><input type="date" value={config.startDate} onChange={e => setConfig({...config, startDate: e.target.value})} /><input type="date" value={config.endDate} onChange={e => setConfig({...config, endDate: e.target.value})} /></div>)}
                  <label className="checkbox-item"><input type="checkbox" checked={modules.wildlifeStream} onChange={() => toggleModule('wildlifeStream')} /><span>Wildlife Trap Stream</span></label>
                  {modules.wildlifeStream && (<div className="sub-config"><input type="text" placeholder="YouTube URL..." value={config.streamUrl} onChange={e => setConfig({...config, streamUrl: e.target.value})} /></div>)}
                </div>
              </div>

              <button className="action-btn" onClick={handleRunAnalysis}>RUN ANALYSIS PIPELINE</button>
              <div className="carousel-section"><h4>Protected Species Intel</h4>{renderInformaticCarousel()}</div>
            </div>
          ) : (
            <div className="upload-wrapper">
              <form className="upload-form" onSubmit={handleUpload}>
                <h3>Submit Observation</h3>
                <div className="group"><label>Species Name (Optional)</label><input type="text" placeholder="e.g. Panthera tigris" value={uploadForm.species} onChange={e => setUploadForm({...uploadForm, species: e.target.value})} /></div>
                <div className="group"><div className="file-drop-area"><input type="file" accept="image/*" onChange={e => setUploadForm({...uploadForm, file: e.target.files[0]})} /><p>{uploadForm.file ? uploadForm.file.name : "Drag & Drop Image Evidence"}</p></div></div>
                <div className="coordinates-display"><p>Location: {coords.lat.toFixed(5)}, {coords.lng.toFixed(5)}</p></div>
                <button type="submit" className="action-btn upload-btn">Upload Data</button>
              </form>
              <div className="uploads-gallery-section">
                <div className="gallery-header"><h4>Recent Uploads</h4>{myUploads.length > 4 && <button className="text-btn" onClick={() => setShowGalleryModal(true)}>View More</button>}</div>
                <div className="gallery-grid">
                  {myUploads.slice(0, 4).map((up) => (<div key={up.id} className="gallery-item"><SecureImage imageId={up.id} alt={up.species} /></div>))}
                  {myUploads.length === 0 && <p className="no-data">No uploads yet.</p>}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="visual-panel">
        {viewState === 'map' && (
          <div className="map-wrapper">
             <button className={`legend-toggle ${showLabels ? 'active' : ''}`} onClick={() => setShowLabels(!showLabels)}>{showLabels ? 'Hide Labels' : 'Show Labels'}</button>
             <MapContainer center={coords} zoom={13} style={{ width: '100%', height: '100%' }}>
               <TileLayer attribution='&copy; Esri' url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" />
               {showLabels && <TileLayer url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}" />}
               <LocationSelector onLocationSelect={handleMapClick} />
               <Marker position={coords} />
               <Circle center={coords} radius={radius * 1000} pathOptions={{ fillColor: '#10b981', fillOpacity: 0.2, color: '#0f2d25', weight: 2 }} />
               
               {/* Map Overlays Logic - Using Base64 */}
               {mapOverlays.landcover && mapOverlays.bounds && (
                 <ImageOverlay 
                   url={mapOverlays.landcover} // Base64 data string
                   bounds={mapOverlays.bounds} 
                   opacity={0.6} 
                 />
               )}
               {mapOverlays.vegetation && mapOverlays.bounds && (
                 <ImageOverlay 
                   url={mapOverlays.vegetation} // Base64 data string
                   bounds={mapOverlays.bounds} 
                   opacity={0.6} 
                 />
               )}
             </MapContainer>
          </div>
        )}

        {viewState === 'grid' && (
          <div className="command-grid-container">
            <div className="grid-header"><h2>Mission Control</h2><button className="back-btn" onClick={() => setViewState('map')}>View Map Overlay</button></div>
            <div className={`grid-matrix ${expandedWidget ? 'has-expanded' : ''}`}>
              {['landcover', 'vegetation', 'uploads', 'panos', 'changeDetection', 'wildlifeStream'].map((type) => (
                <div key={type} className={`grid-box ${expandedWidget === type ? 'expanded' : ''}`} onClick={() => setExpandedWidget(expandedWidget === type ? null : type)}>
                  {renderWidgetContent(type)}
                  {!loading[type] && modules[type] && (<div className="expand-hint">{expandedWidget === type ? 'Collapse' : 'Expand'}</div>)}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {showGalleryModal && (
        <div className="modal-overlay" onClick={() => setShowGalleryModal(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header"><h3>Your Contribution Gallery</h3><button onClick={() => setShowGalleryModal(false)}>×</button></div>
            <div className="full-gallery-grid">
              {myUploads.map((up) => (<div key={up.id} className="gallery-item large"><SecureImage imageId={up.id} alt={up.species} /><div className="img-caption"><span>{up.species || 'Unknown'}</span><small>{new Date().toLocaleDateString()}</small></div></div>))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}