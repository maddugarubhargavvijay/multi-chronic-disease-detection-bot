import React, { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import "./chatbot.css";
import { FaPaperPlane, FaDownload, FaMapMarkerAlt, FaUpload } from "react-icons/fa";

const Chatbot = () => {
  // Chat state
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hello! How can I assist you today?" }
  ]);
  const [input, setInput] = useState("");
  
  // User info and report generation
  const [username, setUsername] = useState(null);
  const [awaitingUsername, setAwaitingUsername] = useState(false);
  const [predictedDisease, setPredictedDisease] = useState(null);
  
  // Doctor selection
  const [doctorOptions, setDoctorOptions] = useState({});
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  
  // Refs for auto-scrolling
  const chatBoxRef = useRef(null);
  const lastMessageRef = useRef(null);

  // API endpoints
  const rasaUrl = "http://127.0.0.1:5005/webhooks/rest/webhook";
  const predictUrl = "http://127.0.0.1:5000/predict";
  const reportUrl = "http://127.0.0.1:5000/generate_report";
  const doctorsUrl = "http://127.0.0.1:5000/get_doctors";
  const selectDoctorUrl = "http://127.0.0.1:5000/select_doctor";

  useEffect(() => {
    lastMessageRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Append a new message to the chat
  const addMessage = (sender, text) => {
    setMessages((prev) => [...prev, { sender, text }]);
  };

  const sendMessage = useCallback(async (message) => {
    if (!message.trim()) return;

    // Always add user message
    addMessage("user", message);
    setInput("");

    // If waiting for username, store it
    if (awaitingUsername) {
      setUsername(message);
      setAwaitingUsername(false);
      addMessage("bot", `Thank you, ${message}! Now, type "generate report" again.`);
      return;
    }

    // Handle report generation command
    if (message.toLowerCase() === "generate report") {
      if (!username) {
          addMessage("bot", "Please enter your name before generating the report.");
          setAwaitingUsername(true);
          return;
      }
      
      try {
        const reportData = {
          user_name: username,
          disease: predictedDisease || "No prediction",
          chat_history: messages,
          doctor: selectedDoctor ? selectedDoctor : null, // Only include if a doctor is selected
        };
        
  
          const response = await axios.post(reportUrl, reportData, { responseType: "blob" });
  
          if (response.status === 200) {
              const blob = new Blob([response.data], { type: "application/pdf" });
              const url = URL.createObjectURL(blob);
  
              // Add message with direct download link
              setMessages((prev) => [
                  ...prev,
                  {
                      sender: "bot",
                      text: (
                          <span>
                               Report generated successfully! Click below to download:
                              <br />
                              <a href={url} download="health_report.pdf" className="download-link">
                                  Download Report
                              </a>
                          </span>
                      ),
                  },
              ]);
          } else {
              addMessage("bot", "Failed to generate report.");
          }
      } catch (error) {
          console.error("Error generating report:", error);
          addMessage("bot", "Could not generate report. Please try again later.");
      }
      return;
  }
    // Handle doctor selection if there are available options and the message is a number
    if (Object.keys(doctorOptions).length > 0) {
      const selectedIndex = parseInt(message);
      if (!isNaN(selectedIndex) && doctorOptions[selectedIndex]) {
        try {
          const response = await axios.post(selectDoctorUrl, { doctor_index: selectedIndex.toString() });
          if (response.data.doctor) {
            setSelectedDoctor(response.data.doctor);
            addMessage(
              "bot",
              ` Doctor Selected:\n ${response.data.doctor.name}\n ${response.data.doctor.location}\nType "generate report" to include this doctor in your PDF.`
            );
          } else {
            addMessage("bot", " Doctor selection failed. Please try again.");
          }
        } catch (error) {
          console.error("Error selecting doctor:", error);
          addMessage("bot", " Failed to select doctor.");
        }
        setDoctorOptions({});
        return;
      } else {
        addMessage("bot", " Invalid selection. Please enter a valid number.");
        return;
      }
    }
    // Default: Send message to Rasa chatbot and display its responses
    try {
      const response = await axios.post(rasaUrl, { sender: "user", message });
      if (response.data.length > 0) {
        response.data.forEach((msg) => addMessage("bot", msg.text));
      }
    } catch (error) {
      console.error("Error:", error);
      addMessage("bot", "Failed to communicate with the chatbot.");
    }
  }, [messages, username, awaitingUsername, doctorOptions, predictedDisease]);

  // Fetch nearby doctors using geolocation
  const fetchDoctors = async () => {
    if (!navigator.geolocation) {
      alert(" Geolocation is not supported by your browser.");
      return;
    }
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;
        addMessage("user", "Searching for nearby doctors...");
        try {
          const response = await axios.post(doctorsUrl, { latitude, longitude });
          if (response.data.doctors && Object.keys(response.data.doctors).length > 0) {
            let doctorListText = "Nearby Doctors:\n";
            Object.entries(response.data.doctors).forEach(([key, doctor]) => {
              doctorListText += `${key}. ${doctor.name} - ${doctor.location}\n`;
            });
            doctorListText += "\n Type a number (e.g., 1) to select a doctor.";
            addMessage("bot", doctorListText);
            setDoctorOptions(response.data.doctors);
          } else {
            addMessage("bot", " No doctors found nearby.");
          }
        } catch (error) {
          console.error("Error:", error);
          addMessage("bot", " Could not fetch doctors. Please try again later.");
        }
      },
      (error) => {
        console.error("Location error:", error);
        addMessage("bot", " Unable to fetch location. Please enable location services.");
      }
    );
  };

  // Upload X-ray file to predict disease
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    // Step 1: Inform user that upload has started
    addMessage("user", " Uploading X-ray, please wait...");
    
    try {
        // Step 2: Inform user that the model is processing
        addMessage("bot", "Processing X-ray... Please wait for the results.");
        
        const response = await axios.post(predictUrl, formData, {
            headers: { "Content-Type": "multipart/form-data" }
        });

        if (response.data && response.data.disease) {
            setPredictedDisease(response.data.disease);
            
            // Step 3: Remove "Processing" message and show results
            addMessage("bot", ` X-ray analysis complete! Disease Detected: **${response.data.disease}**`);
        } else {
            addMessage("bot", " No disease detected. Try another image.");
        }
    } catch (error) {
        console.error("Error predicting disease:", error);
        addMessage("bot", " An error occurred while analyzing the X-ray.");
    }
};

  return (
    <div className="chat-wrapper">
  <h1 className="chat-title">Multi-Chronic Disease Detection Chatbot</h1> 
  <div className="chat-container">
    <div className="chat-box" ref={chatBoxRef}>
      {messages.map((msg, index) => (
        <div
          key={index}
          className={`message ${msg.sender}`}
          ref={index === messages.length - 1 ? lastMessageRef : null}
        >
          {typeof msg.text === "string"
            ? msg.text.split("\n").map((line, i) => (
                <span key={i}>
                  {line}
                  <br />
                </span>
              ))
            : msg.text}
        </div>
      ))}
    </div>
    
    <div className="action-buttons">
      <button onClick={fetchDoctors}>
        <FaMapMarkerAlt /> Find Nearby Doctors
      </button>
      <button onClick={() => sendMessage("generate report")}>
        <FaDownload /> Download Report
      </button>
      <label className="upload-button">
        <FaUpload /> Upload X-ray
        <input type="file" accept="image/*" onChange={handleFileUpload} hidden />
      </label>
    </div>

    <div className="input-container">
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyPress={(e) => {
          if (e.key === "Enter") sendMessage(input);
        }}
        placeholder="Type a message..."
      />
      <button onClick={() => sendMessage(input)}>
        <FaPaperPlane />
      </button>
    </div>
    
    <div class="bubble"></div>
  </div>
</div>

  );
}; 
export default Chatbot;
