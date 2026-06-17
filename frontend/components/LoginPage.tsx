/**
 * Login/Register Page Component with Extended Profile Collection
 */
import { useState, useEffect } from 'react';
import { Loader2, Eye, EyeOff, Moon, Sun, ChevronDown } from 'lucide-react';
import SearchableMultiSelect from './SearchableMultiSelect';
import { useTheme } from '../hooks/useTheme';

const rawApiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
const LOGIN_API_URL = rawApiUrl.endsWith('/') ? rawApiUrl.slice(0, -1) : rawApiUrl;

export interface UserProfile {
  name: string;
  email: string;
  password: string;
  phone: string;
  gender: string;
  dob: string;
  profession: string;
  preferred_deity: string;
  rashi: string;
  gotra: string;
  nakshatra: string;
  favorite_temples: string[];
  past_purchases: string[];
}

interface LoginPageProps {
  onLogin: (email: string, password: string) => Promise<boolean>;
  onRegister: (profile: UserProfile) => Promise<boolean>;
  isLoading: boolean;
  error: string | null;
}

const professionOptions = [
  { value: 'student', label: 'Student' },
  { value: 'professional', label: 'Working Professional' },
  { value: 'business', label: 'Business Owner / Entrepreneur' },
  { value: 'homemaker', label: 'Homemaker' },
  { value: 'retired', label: 'Retired' },
  { value: 'caregiver', label: 'Caregiver' },
  { value: 'other', label: 'Other' },
];

const genderOptions = [
  { value: 'male', label: 'Male' },
  { value: 'female', label: 'Female' },
  { value: 'other', label: 'Other' },
  { value: 'prefer_not_to_say', label: 'Prefer not to say' },
];

const deityOptions = [
  'Shiva', 'Vishnu', 'Brahma', 'Krishna', 'Rama', 'Hanuman', 'Ganesha',
  'Durga', 'Lakshmi', 'Saraswati', 'Parvati', 'Kali', 'Kartikeya',
  'Radha', 'Sita', 'Surya', 'Shani', 'Jagannath', 'Venkateswara',
  'Narasimha', 'Ayyappa', 'Dattatreya', 'Santoshi Mata', 'Nandi',
];

const rashiOptions = [
  { value: 'Mesh', label: 'Mesh (Aries)' },
  { value: 'Vrishabh', label: 'Vrishabh (Taurus)' },
  { value: 'Mithun', label: 'Mithun (Gemini)' },
  { value: 'Kark', label: 'Kark (Cancer)' },
  { value: 'Simha', label: 'Simha (Leo)' },
  { value: 'Kanya', label: 'Kanya (Virgo)' },
  { value: 'Tula', label: 'Tula (Libra)' },
  { value: 'Vrishchik', label: 'Vrishchik (Scorpio)' },
  { value: 'Dhanu', label: 'Dhanu (Sagittarius)' },
  { value: 'Makar', label: 'Makar (Capricorn)' },
  { value: 'Kumbh', label: 'Kumbh (Aquarius)' },
  { value: 'Meen', label: 'Meen (Pisces)' },
];

const gotraOptions = [
  'Bharadwaj', 'Kashyap', 'Vashishtha', 'Vishwamitra', 'Gautam',
  'Jamadagni', 'Atri', 'Agastya', 'Kaushik', 'Sandilya',
  'Parashar', 'Mudgal', 'Garg', 'Katyayan', 'Shandilya', 'Bhrigu',
];

const nakshatraOptions = [
  'Ashwini', 'Bharani', 'Krittika', 'Rohini', 'Mrigashira', 'Ardra',
  'Punarvasu', 'Pushya', 'Ashlesha', 'Magha', 'Purva Phalguni',
  'Uttara Phalguni', 'Hasta', 'Chitra', 'Swati', 'Vishakha',
  'Anuradha', 'Jyeshtha', 'Moola', 'Purva Ashadha', 'Uttara Ashadha',
  'Shravana', 'Dhanishta', 'Shatabhisha', 'Purva Bhadrapada',
  'Uttara Bhadrapada', 'Revati',
];

const templeOptions = [
  'Kashi Vishwanath', 'Tirupati Balaji', 'Vaishno Devi', 'Kedarnath',
  'Badrinath', 'Somnath', 'Rameshwaram', 'Jagannath Puri', 'Dwarkadhish',
  'Golden Temple', 'Shirdi Sai Baba', 'Siddhivinayak', 'Mahakaleshwar',
  'Meenakshi Temple', 'ISKCON Vrindavan', 'Akshardham', 'Lingaraj',
  'Brihadeeswara', 'Padmanabhaswamy', 'Konark Sun Temple', 'Kamakhya',
  'Trimbakeshwar', 'Ujjain Mahakal', 'Amarnath',
];

const selectClass = 'w-full pl-5 pr-12 py-3.5 bg-gray-50/50 dark:bg-gray-800/50 border border-orange-100 dark:border-gray-700 rounded-2xl focus:ring-4 focus:ring-orange-500/5 focus:border-orange-200 dark:focus:border-orange-700 focus:bg-white dark:focus:bg-gray-800 text-sm font-bold text-gray-900 dark:text-gray-100 transition-all outline-none appearance-none cursor-pointer';

// Auto-scroll the focused select into view so the dropdown is visible after click
const handleSelectFocus = (e: React.FocusEvent<HTMLSelectElement>) => {
  const target = e.currentTarget;
  setTimeout(() => target.scrollIntoView({ block: 'center', behavior: 'smooth' }), 50);
};
const inputClass = 'w-full px-5 py-3.5 bg-gray-50/50 dark:bg-gray-800/50 border border-orange-100 dark:border-gray-700 rounded-2xl focus:ring-4 focus:ring-orange-500/5 focus:border-orange-200 dark:focus:border-orange-700 focus:bg-white dark:focus:bg-gray-800 text-sm font-bold text-gray-900 dark:text-gray-100 transition-all outline-none';
const labelClass = 'block text-[10px] font-black uppercase tracking-widest text-gray-400 dark:text-gray-500 mb-1.5 ml-1';

export default function LoginPage({ onLogin, onRegister, isLoading, error }: LoginPageProps) {
  const { theme, toggleTheme } = useTheme();
  const [isRegisterMode, setIsRegisterMode] = useState(false);
  const [registrationStep, setRegistrationStep] = useState(1); // 1: basic info, 2: profile details, 3: spiritual profile

  // Basic info
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  // Profile details
  const [phone, setPhone] = useState('');
  const [gender, setGender] = useState('');
  const [dob, setDob] = useState('');
  const [profession, setProfession] = useState('');

  // Spiritual profile
  const [preferredDeity, setPreferredDeity] = useState<string[]>([]);
  const [rashi, setRashi] = useState('');
  const [gotra, setGotra] = useState('');
  const [nakshatra, setNakshatra] = useState('');
  const [favoriteTemples, setFavoriteTemples] = useState<string[]>([]);
  const [pastPurchases, setPastPurchases] = useState<string[]>([]);

  // Product options fetched from backend
  const [productOptions, setProductOptions] = useState<string[]>([]);

  const [localError, setLocalError] = useState<string | null>(null);

  // Fetch product names from backend
  useEffect(() => {
    fetch(`${LOGIN_API_URL}/api/auth/product-names`)
      .then(res => res.json())
      .then((data: string[]) => setProductOptions(data))
      .catch(() => setProductOptions([]));
  }, []);

  const validateStep1 = (): boolean => {
    if (!name.trim()) {
      setLocalError('Please enter your name');
      return false;
    }
    // Fix 10: Proper email regex validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!email.trim() || !emailRegex.test(email)) {
      setLocalError('Please enter a valid email address');
      return false;
    }
    // Fix 9: Stronger password validation
    if (password.length < 8) {
      setLocalError('Password must be at least 8 characters');
      return false;
    }
    if (!/[a-zA-Z]/.test(password)) {
      setLocalError('Password must contain at least one letter');
      return false;
    }
    if (!/\d/.test(password)) {
      setLocalError('Password must contain at least one number');
      return false;
    }
    if (!/[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\\/~`]/.test(password)) {
      setLocalError('Password must contain at least one special character');
      return false;
    }
    const commonPasswords = ['password', '12345678', 'qwerty', 'abc123', 'letmein', 'welcome', 'admin'];
    if (commonPasswords.some(common => password.toLowerCase().includes(common))) {
      setLocalError('Password is too common — please choose a stronger one');
      return false;
    }
    if (password !== confirmPassword) {
      setLocalError('Passwords do not match');
      return false;
    }
    return true;
  };

  const validateStep2 = (): boolean => {
    // Fix 11: Enforce 10-digit minimum for Indian phone numbers
    const cleanPhone = phone.replace(/\D/g, '');
    if (!cleanPhone || cleanPhone.length < 10 || cleanPhone.length > 15) {
      setLocalError('Please enter a valid phone number (10-15 digits)');
      return false;
    }
    if (!gender) {
      setLocalError('Please select your gender');
      return false;
    }
    if (!dob) {
      setLocalError('Please enter your date of birth');
      return false;
    }
    if (!profession) {
      setLocalError('Please select your profession');
      return false;
    }
    return true;
  };

  const validateStep3 = (): boolean => {
    if (preferredDeity.length === 0) {
      setLocalError('Please select at least one preferred deity');
      return false;
    }
    if (!rashi) {
      setLocalError('Please select your rashi');
      return false;
    }
    if (!gotra) {
      setLocalError('Please select your gotra');
      return false;
    }
    if (!nakshatra) {
      setLocalError('Please select your nakshatra');
      return false;
    }
    if (favoriteTemples.length === 0) {
      setLocalError('Please select at least one temple');
      return false;
    }
    if (pastPurchases.length === 0) {
      setLocalError('Please select at least one past purchase');
      return false;
    }
    return true;
  };

  const handleNextStep = () => {
    setLocalError(null);
    if (registrationStep === 1 && validateStep1()) {
      setRegistrationStep(2);
    } else if (registrationStep === 2 && validateStep2()) {
      setRegistrationStep(3);
    }
  };

  const handleSkipStep = () => {
    setLocalError(null);
    if (registrationStep === 2) {
      setRegistrationStep(3);
    } else if (registrationStep === 3) {
      // Submit with whatever is filled so far
      handleSubmitWithDefaults();
    }
  };

  const handleSubmitWithDefaults = async () => {
    await onRegister({
      name,
      email,
      password,
      phone: phone || '',
      gender: gender || '',
      dob: dob || '',
      profession: profession || '',
      preferred_deity: preferredDeity.length > 0 ? preferredDeity.join(', ') : '',
      rashi: rashi || '',
      gotra: gotra || '',
      nakshatra: nakshatra || '',
      favorite_temples: favoriteTemples,
      past_purchases: pastPurchases,
    });
  };

  const handlePrevStep = () => {
    setLocalError(null);
    setRegistrationStep(registrationStep - 1);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);

    if (isRegisterMode) {
      if (registrationStep < 3) {
        handleNextStep();
        return;
      }

      // Step 3 - Final registration
      if (!validateStep3()) {
        return;
      }

      await onRegister({
        name,
        email,
        password,
        phone,
        gender,
        dob,
        profession,
        preferred_deity: preferredDeity.join(', '),
        rashi,
        gotra,
        nakshatra,
        favorite_temples: favoriteTemples,
        past_purchases: pastPurchases,
      });
    } else {
      await onLogin(email, password);
    }
  };

  const switchMode = () => {
    setIsRegisterMode(!isRegisterMode);
    setRegistrationStep(1);
    setLocalError(null);
    setName('');
    setConfirmPassword('');
    setPhone('');
    setGender('');
    setDob('');
    setProfession('');
    setPreferredDeity([]);
    setRashi('');
    setGotra('');
    setNakshatra('');
    setFavoriteTemples([]);
    setPastPurchases([]);
  };

  // Calculate max date for DOB (user should be at least 13 years old)
  const today = new Date();
  const maxDate = new Date(today.getFullYear() - 13, today.getMonth(), today.getDate())
    .toISOString().split('T')[0];
  const minDate = new Date(today.getFullYear() - 100, today.getMonth(), today.getDate())
    .toISOString().split('T')[0];

  return (
    <div className="min-h-[100dvh] bg-gradient-to-br from-orange-50/50 via-white to-amber-50/50 dark:from-gray-950 dark:via-[#0a0a0a] dark:to-gray-950 flex items-center justify-center p-6 font-['Inter'] relative">
      <button
        onClick={toggleTheme}
        className="absolute top-6 right-6 p-2.5 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border border-orange-100 dark:border-gray-700 rounded-xl shadow-sm hover:shadow-md transition-all text-gray-600 dark:text-gray-400 hover:text-orange-600 dark:hover:text-orange-400 z-10"
        aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
      >
        {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
      </button>
      <div className="w-full max-w-[400px] animate-fade-in">
        {/* Logo Text */}
        <div className="text-center mb-8 flex flex-col items-center">
          <img src={theme === 'dark' ? '/logo-full-dark.png' : '/logo-full.png'} alt="3ioNetra" className="h-20 object-contain mb-2 dark:drop-shadow-[0_0_15px_rgba(255,255,255,0.12)]" />
          <p className="text-[11px] font-black uppercase tracking-[0.3em] text-orange-600">Elevate Your Spirit</p>
        </div>

        {/* Login/Register Form */}
        <div data-testid="auth-form" className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl rounded-[2rem] shadow-2xl shadow-orange-900/5 dark:shadow-black/20 p-7 md:p-8 border border-orange-100/50 dark:border-gray-800">
          <h2 data-testid="auth-heading" className="text-base font-black text-gray-900 dark:text-gray-100 mb-1 text-center uppercase tracking-widest opacity-80">
            {isRegisterMode
              ? (registrationStep === 1 ? 'Know Your Bhakt' : registrationStep === 2 ? 'Complete Profile' : 'Spiritual Profile')
              : 'Welcome Back'}
          </h2>
          {isRegisterMode && registrationStep > 1 && (
            <p className="text-[9px] font-bold text-gray-400 dark:text-gray-500 text-center mb-4 uppercase tracking-wider">Optional &mdash; you can skip this step</p>
          )}
          {isRegisterMode && registrationStep === 1 && <div className="mb-5" />}

          {/* Progress indicator for registration */}
          {isRegisterMode && (
            <div data-testid="step-indicators" className="flex items-center justify-center gap-1.5 mb-6">
              {[1, 2, 3].map(step => (
                <div key={step} className="flex items-center gap-1.5">
                  <div
                    aria-label={`Step ${step} of 3`}
                    aria-current={registrationStep === step ? 'step' : undefined}
                    className={`flex items-center justify-center w-6 h-6 rounded-full text-[9px] font-black transition-all duration-500 ${
                      registrationStep > step ? 'bg-orange-500 text-white' :
                      registrationStep === step ? 'bg-orange-500 text-white ring-4 ring-orange-500/20' :
                      'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400'
                    }`}
                  >
                    {registrationStep > step ? '\u2713' : step}
                  </div>
                  {step < 3 && <div className={`w-8 h-0.5 rounded-full transition-all duration-500 ${registrationStep > step ? 'bg-orange-500' : 'bg-gray-200 dark:bg-gray-700'}`} />}
                </div>
              ))}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Step 1: Basic Info */}
            {(!isRegisterMode || registrationStep === 1) && (
              <>
                {isRegisterMode && (
                  <div>
                    <label htmlFor="name" className={labelClass}>Full Name</label>
                    <input
                      id="name"
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="Your Name"
                      required={isRegisterMode}
                      disabled={isLoading}
                      className={inputClass}
                    />
                  </div>
                )}

                <div>
                  <label htmlFor="email" className={labelClass}>Email</label>
                  <input
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="name@nexus.com"
                    required
                    disabled={isLoading}
                    className={inputClass}
                  />
                </div>

                <div>
                  <label htmlFor="password" className={labelClass}>Password</label>
                  <div className="relative">
                    <input
                      id="password"
                      type={showPassword ? 'text' : 'password'}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="••••••••"
                      required
                      disabled={isLoading}
                      className={`${inputClass} pr-14`}
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      aria-label={showPassword ? 'Hide password' : 'Show password'}
                      className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 dark:text-gray-500 hover:text-orange-600 dark:hover:text-orange-400 transition-colors"
                    >
                      {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                    </button>
                  </div>
                </div>

                {isRegisterMode && (
                  <div>
                    <label htmlFor="confirmPassword" className={labelClass}>Confirm</label>
                    <input
                      id="confirmPassword"
                      type={showPassword ? 'text' : 'password'}
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      placeholder="••••••••"
                      disabled={isLoading}
                      required={isRegisterMode}
                      className={inputClass}
                    />
                  </div>
                )}
              </>
            )}

            {/* Step 2: Profile Details */}
            {isRegisterMode && registrationStep === 2 && (
              <>
                <div>
                  <label htmlFor="phone" className={labelClass}>Phone</label>
                  <input
                    id="phone"
                    type="tel"
                    value={phone}
                    onChange={(e) => setPhone(e.target.value.replace(/\D/g, '').slice(0, 15))}
                    placeholder="Your Phone Number"
                    required
                    className={inputClass}
                  />
                </div>

                <div>
                  <label htmlFor="gender" className={labelClass}>Gender</label>
                  <div className="relative">
                    <select id="gender" value={gender} onChange={(e) => setGender(e.target.value)} onFocus={handleSelectFocus} required className={selectClass}>
                      <option value="">Select Gender</option>
                      {genderOptions.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                    </select>
                    <ChevronDown className="w-4 h-4 text-gray-400 dark:text-gray-500 absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none" />
                  </div>
                </div>

                <div>
                  <label htmlFor="dob" className={labelClass}>Date of Birth</label>
                  <input id="dob" type="date" value={dob} onChange={(e) => setDob(e.target.value)} max={maxDate} min={minDate} required className={`${inputClass} cursor-pointer`} />
                </div>

                <div>
                  <label htmlFor="profession" className={labelClass}>Profession</label>
                  <div className="relative">
                    <select id="profession" value={profession} onChange={(e) => setProfession(e.target.value)} onFocus={handleSelectFocus} required className={selectClass}>
                      <option value="">Select Profession</option>
                      {professionOptions.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                    </select>
                    <ChevronDown className="w-4 h-4 text-gray-400 dark:text-gray-500 absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none" />
                  </div>
                </div>
              </>
            )}

            {/* Step 3: Spiritual Profile */}
            {isRegisterMode && registrationStep === 3 && (
              <div className="space-y-4 max-h-[55vh] overflow-y-auto pr-1">
                {/* Preferred Deity - searchable multi-select */}
                <SearchableMultiSelect
                  label="Preferred Deity"
                  options={deityOptions}
                  selected={preferredDeity}
                  onChange={setPreferredDeity}
                  placeholder="Search deities..."
                />

                {/* Rashi - single dropdown */}
                <div>
                  <label htmlFor="rashi" className={labelClass}>Rashi (Zodiac)</label>
                  <div className="relative">
                    <select id="rashi" value={rashi} onChange={(e) => setRashi(e.target.value)} onFocus={handleSelectFocus} required className={selectClass}>
                      <option value="">Select Rashi</option>
                      {rashiOptions.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                    </select>
                    <ChevronDown className="w-4 h-4 text-gray-400 dark:text-gray-500 absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none" />
                  </div>
                </div>

                {/* Gotra - single dropdown */}
                <div>
                  <label htmlFor="gotra" className={labelClass}>Gotra</label>
                  <div className="relative">
                    <select id="gotra" value={gotra} onChange={(e) => setGotra(e.target.value)} onFocus={handleSelectFocus} required className={selectClass}>
                      <option value="">Select Gotra</option>
                      {gotraOptions.map((g) => <option key={g} value={g}>{g}</option>)}
                    </select>
                    <ChevronDown className="w-4 h-4 text-gray-400 dark:text-gray-500 absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none" />
                  </div>
                </div>

                {/* Nakshatra - single dropdown */}
                <div>
                  <label htmlFor="nakshatra" className={labelClass}>Nakshatra</label>
                  <div className="relative">
                    <select id="nakshatra" value={nakshatra} onChange={(e) => setNakshatra(e.target.value)} onFocus={handleSelectFocus} required className={selectClass}>
                      <option value="">Select Nakshatra</option>
                      {nakshatraOptions.map((n) => <option key={n} value={n}>{n}</option>)}
                    </select>
                    <ChevronDown className="w-4 h-4 text-gray-400 dark:text-gray-500 absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none" />
                  </div>
                </div>

                {/* Temples - searchable multi-select */}
                <SearchableMultiSelect
                  label="Temples Visited"
                  options={templeOptions}
                  selected={favoriteTemples}
                  onChange={setFavoriteTemples}
                  placeholder="Search temples..."
                />

                {/* Past Purchases - searchable multi-select from DB */}
                <SearchableMultiSelect
                  label="Past Spiritual Purchases"
                  options={productOptions}
                  selected={pastPurchases}
                  onChange={setPastPurchases}
                  placeholder="Search products..."
                  loading={productOptions.length === 0}
                />
              </div>
            )}

            {/* Error Display */}
            {(error || localError) && (
              <div className="bg-red-50/50 dark:bg-red-900/20 border border-red-100 dark:border-red-800 rounded-2xl p-4 animate-fade-in">
                <p className="text-[11px] font-black uppercase tracking-tighter text-red-600 dark:text-red-400 text-center">{localError || error}</p>
              </div>
            )}

            {/* Buttons */}
            <div className="flex gap-3 pt-4">
              {isRegisterMode && registrationStep > 1 && (
                <button
                  type="button"
                  onClick={handlePrevStep}
                  className="flex-1 py-4 bg-white dark:bg-gray-800 border border-orange-100 dark:border-gray-700 text-orange-600 dark:text-orange-400 font-black uppercase tracking-widest text-[10px] rounded-2xl hover:bg-orange-50 dark:hover:bg-gray-700 transition-all active:scale-95 shadow-sm"
                >
                  Back
                </button>
              )}

              <button
                type="submit"
                disabled={isLoading}
                className="flex-[2] py-4 bg-gradient-to-br from-orange-500 to-amber-600 text-white font-black uppercase tracking-[0.2em] text-[10px] rounded-2xl hover:from-orange-600 hover:to-amber-700 transition-all disabled:opacity-50 disabled:grayscale shadow-lg shadow-orange-900/10 active:scale-95 flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : isRegisterMode ? (
                  registrationStep < 3 ? 'Next Step' : 'Know Your Bhakt'
                ) : (
                  'Sign In'
                )}
              </button>

              {isRegisterMode && registrationStep > 1 && (
                <button
                  type="button"
                  onClick={handleSkipStep}
                  disabled={isLoading}
                  className="flex-1 py-4 bg-white dark:bg-gray-800 border border-dashed border-gray-200 dark:border-gray-700 text-gray-400 dark:text-gray-500 font-black uppercase tracking-widest text-[10px] rounded-2xl hover:text-orange-600 dark:hover:text-orange-400 hover:border-orange-200 dark:hover:border-orange-700 transition-all active:scale-95"
                >
                  Skip
                </button>
              )}
            </div>
          </form>

          <div className="mt-8 text-center">
            <p className="text-[11px] font-bold text-gray-400 dark:text-gray-500 uppercase tracking-widest">
              {isRegisterMode ? 'Already a member?' : "New here?"}
              <button
                onClick={switchMode}
                className="ml-2 text-orange-600 dark:text-orange-400 hover:text-orange-700 dark:hover:text-orange-300 transition-colors underline underline-offset-4"
              >
                {isRegisterMode ? 'Sign In' : 'Know Your Bhakt'}
              </button>
            </p>
          </div>
        </div>

        {/* Footer */}
        <p className="text-center text-[9px] font-black uppercase tracking-[0.3em] text-gray-300 dark:text-gray-600 mt-10 opacity-60">
          Nexus of Ancient Wisdom & Modern Soul
        </p>
      </div>
    </div>
  );
}
