import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'
import WebCamComponent from './webcam' 

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <WebCamComponent apiBase = "https://main-cont-543056702319.asia-southeast1.run.app" captureInterval = {1000} />
  </React.StrictMode>,
)
