// webcam component for capturing screenshots from webcam + obtaining user screen dimensions + feeding into fastAPI backend via multipart forms

import React, { useRef, useCallback, useEffect } from "react";
import Webcam from "react-webcam";

// prop declaration for type safety later on 
// props are waht's fed into our COMPONENT
type Props = {
    apiBase: string
    captureInterval: number
    };

// initialise the webcam FUNCTIONAL COMPONENT that takes in prop
// this component does the following things:

// 1. access webcam using react-webcam
// 2. define capture function that sends frames to backend AND obtains predicted screen coordinates
// 3. obtain screen coords at regular interval using .setInterval and capture
// 4. perform scrolling motion at every interval too
// 5. return a JSX element to render in DOM by React


const webCamComponent: React.FC<Props> = ({
    apiBase,
    captureInterval
}) => {
    const webCamRef = useRef<Webcam>(null);

    const capture = useCallback(async () => {
        const ss = webCamRef.current?.getScreenshot();
        if (!ss) return;

        const res = await fetch(ss);
        const blob = await res.blob();

        const formData = new FormData();
        const screenHeight = window.innerHeight;
        formData.append('image', blob);
        formData.append('screenHeight', screenHeight.toString());

        const response = await fetch(`${apiBase}/upload`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            return console.error("Failed to send image to endpoint");
        }

        const [x, y] = await response.json(); // pauses here, added to stack 

        // scrolling mechanism
        if (y > screenHeight * 0.6){
            window.scrollBy({top: 150, behavior: 'smooth'});
        } else if (y < screenHeight * 0.4) {
            window.scrollBy({top: -150, behavior: 'smooth'})
        };
        },[apiBase, webCamRef]);

    const timerRef = useRef<number | null>(null);
    useEffect(() => {
        timerRef.current = window.setInterval(capture, captureInterval);
        return () => {
            if (timerRef.current !== null) {
                window.clearInterval(timerRef.current);
            }
        };
    }, [capture, captureInterval]);

    // react functional components have to return JSX
    // i.e code that'll be rendered in the DOM/browser
    return (
        <div>
            <Webcam
                ref={webCamRef}
                screenshotFormat="image/jpeg"
                audio={false}
                className="rounded-xl shadow-lg"
            />
            {/* Scrollable content */}
            <div style={{ height: "2000px", padding: "2rem" }}>
                <h1>Scrollable Demo Content</h1>
                <p>
                    {Array.from({ length: 200 }).map((_, i) => (
                        <span key={i}>Line {i + 1}<br /></span>
                    ))}
                </p>
            </div>
        </div>
    );
}

export default webCamComponent;