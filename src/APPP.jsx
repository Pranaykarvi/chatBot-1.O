import Chatbot from "./Chatbox.jsx";
import "./style.css";

export default function App() {
  return (
    <div
      className="App"
      style={{ height: "98vh", display: "flex", flexDirection: "column" }}
    >
      <div>
        <h3>Deep learning chatbot</h3>
      </div>
      <div
        style={{
          padding: "15px",
          height: "100%",
          border: "3px solid black",
          overflow: "scroll"
        }}
      >
        <Chatbot />
      </div>
    </div>
  );
}
