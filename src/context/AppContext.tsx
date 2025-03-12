import React, { createContext, useContext, useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';

interface AppContextType {
    backend: string;
    userID: string | null;
    loginStatus: "logged out" | "logging in" | "logged in";
    handleUserLogin: (enteredUserID: string, password: string) => void;
    addUser: (userID: string, password: string) => void;
    admins: string[];
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ backend: string; children: React.ReactNode }> = ({ backend, children }) => {
    const [userID, setUserID] = useState<string | null>(null);
    const [loginStatus, setLoginStatus] = useState<"logged out" | "logging in" | "logged in">("logging in");
    const [attemptedQuickLogin, setAttemptedQuickLogin] = useState(false);
    const [searchParams] = useSearchParams();
    const userParam = searchParams.get('user');
    const canvasParam = searchParams.get('canvas');

    const admins = ["shm", "elaine", "ethan", "sophia", "bob"];

    const addUser = async (userID: string, password: string) => {
        try {
            const response = await fetch(`${backend}/api/add-user`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ userID, password })
            });
            const data = await response.json();
            if (response.ok) {
                console.log(`User added: ${userID}`);
            } else {
                console.error(`Error adding user: ${data.error}`);
            }
        } catch (error) {
            console.error('Error adding user:', error);
        }
    };

    const handleUserLogin = async (enteredUserID: string, password?: string) => {
        //logs in the user in the url, or sends us to the url to log you in on the next load
        password = password || "";
        if (enteredUserID === ""){
            setLoginStatus("logged out")
            return
        }
        if (!userParam) {
            const newUrl = new URL(window.location.href);
            newUrl.searchParams.set('user', enteredUserID);
            if (canvasParam) {
                newUrl.searchParams.set('canvas', canvasParam);
            }
            window.location.href = newUrl.toString();
            return;
        }
        try {
            const response = await fetch(`${backend}/api/list-users`);
            const data = await response.json();
        
            if (response.ok) {
                const userExists = data.users.some((user: { userId: string }) => user.userId === userParam);
                if (userExists) {
                    setUserID(userParam);
                    console.log(`User logged in: ${userParam}...`);

                    localStorage.setItem("last_userID", userParam);
                    localStorage.setItem("last_password", password);

                    setLoginStatus("logged in");
                    
                
                } else {
                    console.error(`User does not exist: ${userParam}`);
                    //setLoginStatus("logged out");
                    const newUrl = new URL(window.location.href);
                    newUrl.searchParams.delete('user');
                    window.location.href = newUrl.toString();
                }
            } else {
                console.error(`Error fetching users: ${data.error}`);
                setLoginStatus("logged out");
            }
        } catch (error) {
            console.error('Error logging in user:', error);
            setLoginStatus("logged out");
        }
    };

    useEffect(() => { //LOGIN... 
        if (!userParam){
            console.log("no user param, operating in logged-out mode.")
            setLoginStatus("logged out");
        }
        if (userParam && !attemptedQuickLogin) { //there's a user id in the url, let's try loggin in once
            console.log("Attempting login from url: " + userParam);
            handleUserLogin(userParam);
        } 
        // else if (!attemptedQuickLogin) { //there's no user id in the url, let's try loggin in from the browser once
        //     console.log("Attempting quick login from browser...");
        //     const storedUserID = localStorage.getItem("last_userID");
        //     const storedPassword = localStorage.getItem("last_password");
        //     if (storedUserID && storedPassword !== null) {
        //         handleUserLogin(storedUserID, storedPassword);
        //         console.log(`Quick login with userID: ${storedUserID}`);
        //     } else {
        //         console.log(`FAILED to find a browser-stored UserID: ${storedUserID}, Stored Password: ${storedPassword}`);
        //         setLoginStatus("logged out");
        //     }
            
        // }
        //there's no valid user id in the url or in the browser
        setAttemptedQuickLogin(true);

        console.log(`Connecting to ${backend.includes("local") ? "local" : backend} backend`);
    }, [backend, userID, userParam, canvasParam]);

    return (
        <AppContext.Provider value={{ backend, userID, loginStatus, handleUserLogin, addUser, admins }}>
            {children}
        </AppContext.Provider>
    );
};

export const useAppContext = (): AppContextType => {
    const context = useContext(AppContext);
    if (!context) {
        throw new Error('useAppContext must be used within an AppProvider');
    }
    return context;
};
