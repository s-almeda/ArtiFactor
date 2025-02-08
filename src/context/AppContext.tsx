import React, { createContext, useContext, useState, useEffect } from 'react';

interface AppContextType {
    backend: string;
    loadCanvasRequest: boolean;
    setLoadCanvasRequest: React.Dispatch<React.SetStateAction<boolean>>;

    userID: string | null;
    handleUserLogin: (enteredUserID: string) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ backend: string; children: React.ReactNode }> = ({ backend, children }) => {
    const [loadCanvasRequest, setLoadCanvasRequest] = useState(false);
    const [userID, setUserID] = useState<string | null>("default");

    const handleUserLogin = async (enteredUserID: string) => {
        try {
            const response = await fetch(`${backend}/api/list-users`);
            const data = await response.json();

            if (response.ok) {
                const userExists = data.users.some((user: { id: string }) => user.id === enteredUserID);

                if (!userExists) {
                    const addUserResponse = await fetch('/api/add-user', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ userID: enteredUserID, password: "" }),
                    });

                    if (addUserResponse.ok) {
                        console.log(`User added: ${enteredUserID}`);
                    } else {
                        const addUserData = await addUserResponse.json();
                        console.error(`Error adding user: ${addUserData.error}`);
                    }
                } else {
                    console.log(`User already exists: ${enteredUserID}`);
                }

                setUserID(enteredUserID);
                console.log(`User logged in: ${enteredUserID}`);
            } else {
                console.error(`Error fetching users: ${data.error}`);
            }
        } catch (error) {
            console.error('Error logging in user:', error);
        }
    };

    useEffect(() => {
        console.log(`Initializing App with backend: ${backend} for userID: ${userID}`);
    }, [backend, userID]);

    return (
        <AppContext.Provider value={{ backend, loadCanvasRequest, setLoadCanvasRequest, userID, handleUserLogin }}>
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