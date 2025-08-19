// Import React and its useful hooks
import React, { useState, useEffect, useRef, useCallback } from "react";

// Import axios for making HTTP requests
import axios from "axios";

// Import styles and icons
import "./chatbot.css";
import { FaIcons } from "react-icons/fa";

// Main Chatbot component function
function Chatbot() {
  // States to store various values
  const [messages, setMessages] = useState([{ sender: "bot", text: "Hello! I am your health assistant. How can I help you today?" }]); // stores all chat messages
  const [input, setInput] = useState(""); // stores user input text
  const [username, setUsername] = useState(null); // stores user's name
  const [awaitingUsername, setAwaitingUsername] = useState(false); // whether bot is waiting for user's name
  const [predictedDisease, setPredictedDisease] = useState(null); // predicted disease after X-ray upload
  const [doctorOptions, setDoctorOptions] = useState({}); // nearby doctor options from backend
  const [selectedDoctor, setSelectedDoctor] = useState(null); // selected doctor from user

  // References to DOM elements for scrolling
  const chatBoxRef = useRef(null); // ref to chat box
  const lastMessageRef = useRef(null); // ref to last message for auto scroll

  // URLs to interact with backend APIs
  const rasaUrl = "http://127.0.0.1:5005/webhooks/rest/webhook"; // URL for chatbot
  const predictUrl = "http://127.0.0.1:5000/predict"; // URL to send X-ray image
  const reportUrl = "http://127.0.0.1:5000/generate_report"; // URL to generate PDF report
  const doctorsUrl = "http://127.0.0.1:5000/get_doctors"; // URL to get nearby doctors
  const selectDoctorUrl = "http://127.0.0.1:5000/select_doctor"; // URL to confirm doctor selection

  // Auto-scroll to the last chat message
  useEffect(() => {
    lastMessageRef.current?.scrollIntoView({ behavior: "smooth" }); // scroll smoothly
  }, [messages]);

  // Function to add a new message to chat
  const addMessage = (sender, text) => {
    setMessages((prevMessages) => [...prevMessages, { sender, text }]); // add new message to message array
  };

  // Function to send message to chatbot and handle response
  const sendMessage = async () => {
    if (!input.trim()) return; // if input is empty, do nothing
    const message = input;
    setInput(""); // clear input field
    addMessage("user", message); // show user message

    // Handle report generation
    if (message.toLowerCase().includes("generate report")) {
      try {
        const response = await axios.post(reportUrl, {
          username,
          disease: predictedDisease,
          doctor: selectedDoctor,
        }, {
          responseType: "blob",
        });

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute("download", "report.pdf");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        addMessage("bot", "Report downloaded successfully."); // show success
      } catch (error) {
        console.error("Error generating report:", error);
        addMessage("bot", "Failed to generate report."); // show failure
      }
      return;
    }

    // Save username if asked earlier
    if (awaitingUsername) {
      setUsername(message);
      setAwaitingUsername(false);
      addMessage("bot", `Thanks, ${message}. How can I assist you today?`);
      return;
    }

    // Check if user selected doctor option
    if (doctorOptions && Object.keys(doctorOptions).includes(message)) {
      const doctor = doctorOptions[message];
      setSelectedDoctor(doctor);
      try {
        await axios.post(selectDoctorUrl, { doctor });
        addMessage("bot", `You selected ${doctor.name}, ${doctor.specialization}.`);
      } catch (error) {
        console.error("Error selecting doctor:", error);
        addMessage("bot", "Failed to select doctor.");
      }
      return;
    }

    // Normal chatbot message
    try {
      const response = await axios.post(rasaUrl, {
        sender: username || "user",
        message,
      });
      response.data.forEach((msg) => {
        addMessage("bot", msg.text);
        if (msg.text.includes("your name")) {
          setAwaitingUsername(true); // bot is asking for name
        }
      });
    } catch (error) {
      console.error("Error sending message:", error);
      addMessage("bot", "Something went wrong. Please try again later.");
    }
  };

  // Function to get user's current location and find doctors
  const fetchDoctors = () => {
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        try {
          const { latitude, longitude } = position.coords;
          const response = await axios.post(doctorsUrl, { latitude, longitude });
          const doctors = response.data;
          setDoctorOptions(doctors);

          // Show list of doctors in chat
          Object.entries(doctors).forEach(([key, doctor]) => {
            addMessage("bot", `${key}. ${doctor.name}, ${doctor.specialization}, ${doctor.location}`);
          });
          addMessage("bot", "Please type the number of the doctor you would like to select.");
        } catch (error) {
          console.error("Error fetching doctors:", error);
          addMessage("bot", "Failed to fetch doctors.");
        }
      },
      (error) => {
        console.error("Error getting location:", error);
        addMessage("bot", "Failed to get your location. Please allow location access.");
      }
    );
  };

  // Function to handle image upload for prediction
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    addMessage("user", `Uploaded file: ${file.name}`);
    addMessage("bot", "Analyzing image...");

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await axios.post(predictUrl, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const result = response.data.prediction;
      setPredictedDisease(result);
      addMessage("bot", `Predicted disease: ${result}`);
    } catch (error) {
      console.error("Error uploading file:", error);
      addMessage("bot", "Failed to analyze image.");
    }
  };

  // HTML and chat UI rendering
  return (
    <div className="chatbot-container">
      <div className="chat-box" ref={chatBoxRef}>
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`message ${msg.sender === "user" ? "user-message" : "bot-message"}`}
            ref={index === messages.length - 1 ? lastMessageRef : null}
          >
            {msg.text}
          </div>
        ))}
      </div>

      <div className="action-buttons">
        <button onClick={fetchDoctors}>Find Nearby Doctors</button>
        <label className="custom-file-upload">
          <input type="file" onChange={handleFileUpload} /> Upload X-ray
        </label>
        <button onClick={() => sendMessage("generate report")}>Generate Report</button>
      </div>

      <div className="input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Type your message..."
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

// Exporting Chatbot component to be used in the app
export default Chatbot;
