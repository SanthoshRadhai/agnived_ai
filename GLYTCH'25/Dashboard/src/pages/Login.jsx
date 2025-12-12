import { useState, useEffect } from 'react';
import axios from 'axios';
import '../styles/Login.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

const ECO_FACTS = [
  "The Amazon Rainforest produces 20% of the world's oxygen.",
  "One hectare of mangrove forest absorbs more carbon than one hectare of tropical rainforest.",
  "Pangolins are the most trafficked mammals in the world, vital for soil aeration.",
  "Bees pollinate 70 of the around 100 crop species that feed 90% of the world.",
  "Sea otters wrap themselves in kelp to keep from floating away while sleeping.",
  "Fungi are the 'internet' of the forest, connecting trees via mycelial networks.",
  "Coral reefs cover less than 1% of the ocean but support 25% of all marine life.",
  "Elephants are 'ecosystem engineers'; their paths act as firebreaks and rain channels.",
  "A single bat can eat up to 1,200 mosquito-sized insects every hour.",
  "Sharks play a vital role in keeping ocean ecosystems healthy by removing the weak and sick.",
  "Wetlands act as natural water filters, removing pollutants before they reach the ocean.",
  "The dung beetle is the only insect known to navigate using the Milky Way.",
  "Trees communicate distress signals to other trees through airborne chemical signals.",
  "Whale poop fertilizes the ocean, promoting phytoplankton growth which absorbs CO2.",
  "Peatlands store twice as much carbon as all the world's forests combined.",
  "Tigers are a keystone species; their presence indicates a healthy forest ecosystem.",
  "Snow Leopards regulate the population of herbivores, preventing overgrazing in the Himalayas.",
  "Ants make up roughly 20% of all terrestrial living animal biomass.",
  "Seagrass meadows can capture carbon up to 35 times faster than tropical rainforests.",
  "Vultures prevent the spread of disease by consuming carrion that would otherwise rot."
];

function Login({ onLoginSuccess }) {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({ userid: '', name: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  // Carousel State
  const [activeFacts, setActiveFacts] = useState([]);
  const [currentFactIndex, setCurrentFactIndex] = useState(0);

  useEffect(() => {
    const shuffled = [...ECO_FACTS].sort(() => 0.5 - Math.random());
    setActiveFacts(shuffled.slice(0, 5));
  }, []);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentFactIndex((prev) => (prev + 1) % 5);
    }, 6000);
    return () => clearInterval(timer);
  }, [activeFacts]);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (isLogin) {
        const response = await axios.post(`${API_URL}/auth/login`, {
          userid: formData.userid,
          password: formData.password,
        });
        localStorage.setItem('token', response.data.access_token);
        localStorage.setItem('userid', response.data.userid);
        localStorage.setItem('role', response.data.role);
        onLoginSuccess(response.data);
      } else {
        await axios.post(`${API_URL}/auth/register`, {
          userid: formData.userid,
          name: formData.name,
          password: formData.password,
        });
        setIsLogin(true);
        setError('Registration successful! Please login.');
        setFormData({ ...formData, password: '' });
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page-container">
      {/* LEFT SIDE: Visuals */}
      <div className="login-visual-section">
        <div className="visual-overlay">
          <div className="brand-header">
            <span className="logo-icon">🌿</span>
            <h1>AgniVed</h1>
            <p className="tagline">Geospatial & Wildlife Analysis</p>
          </div>

          <div className="fact-carousel-container">
            <p className="fact-label">DID YOU KNOW?</p>
            <div className="fact-text-area">
              {activeFacts.length > 0 && (
                <p key={currentFactIndex} className="fact-text">
                  "{activeFacts[currentFactIndex]}"
                </p>
              )}
            </div>
            <div className="carousel-dots">
              {activeFacts.map((_, idx) => (
                <span 
                  key={idx} 
                  className={`dot ${idx === currentFactIndex ? 'active' : ''}`}
                  onClick={() => setCurrentFactIndex(idx)}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* RIGHT SIDE: Form */}
      <div className="login-form-section">
        <div className="form-box">
          <div className="form-intro">
            <h2>{isLogin ? 'Welcome Back' : 'Join the Initiative'}</h2>
            <p>
              {isLogin 
                ? 'Enter your credentials to access the dashboard.' 
                : 'Create an account to start monitoring biodiversity.'}
            </p>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="input-group">
              <label>Username</label>
              <input
                type="text"
                name="userid"
                value={formData.userid}
                onChange={handleChange}
                placeholder="e.g. ranger_01"
                required
              />
            </div>

            {!isLogin && (
              <div className="input-group">
                <label>Full Name</label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  placeholder="John Doe"
                  required
                />
              </div>
            )}

            <div className="input-group">
              <label>Password</label>
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="••••••••"
                required
              />
            </div>

            {error && (
              <div className={`status-message ${error.includes('successful') ? 'success' : 'error'}`}>
                {error}
              </div>
            )}

            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? 'Processing...' : (isLogin ? 'Sign In' : 'Create Account')}
            </button>
          </form>

          <div className="auth-switch">
            <p>
              {isLogin ? "New to AgniVed? " : "Already have an account? "}
              <button 
                className="switch-btn"
                onClick={() => { setIsLogin(!isLogin); setError(''); }}
              >
                {isLogin ? 'Create an account' : 'Log in here'}
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Login;