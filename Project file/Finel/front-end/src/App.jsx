import { useState } from 'react'
import Land from './Land'
import Combined from './Combined'
import './styles.css';  // Update the path if the file is in a different folder
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <Land/>
      <h1>===================================</h1>
      <Combined/>
    </>
  )
}

export default App
