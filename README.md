FYI: don't pull request this branch! I'm making it just to have a version of the app as demoed on February 5th available in vercel lol

# ArtiFactor! 
(name is tentative lol)

## HOW TO RUN THIS CODE
1. git clone this
2. open up Terminal and cd into the directory!
### Frontend Developing...
- cd into the src folder
- run "npm run dev"
- go visit https://localhost:5173 with your web browser!
#### What to edit...
- If you want to edit the overall app structure (e.g. the size of the canvas or sidebar) Start in App.tsx
- If you want to edit stuff that exists in the infinite canvas, start in Flow.tsx
- - different kinds of node component are each defined in the /nodes folder, so here is where you could add a new kind of node
  - each type of node is defined in types.ts
  - the nodes that the application "starts" with are defined under index.ts
  - we're mostly using the ImageNode, TextNode, and T2IGenerator Node rn. the other ones are from the documentation/are prototypes for the ones that exist now.
- The About page is About.tsx
- The left Sidebar component is in Sidebar.tsx
### PUSHING YOUR CHANGES
go ahead and make branches / pull requests to your heart's desire! If you have a change that makes sense to just push to main right away I trust you! but please feel very encouraged to communicate / ask questions.
To actually update the frontend that is showing up on Vercel, you have to run ``npm run build``
Once you do this, it updates the version of the app in the "dist" folder. This is what vercel is actually watching for before rebuilding the site. You'll have to resolve all errors, even tiny ones, before it lets you build. 


### Backend Developing...
1. (assuming you've already git cloned this repo) cd into the /backend folder
2. run the backend server with ``node server.js`` -- it should start running on localhost:3000
3. That's cool and all, but the frontend will still be sending requests to my snailbunny.site domain, which has an online version of the server. To test your changes locally, go to lines 21-27 of Flow.tsx where it says:
        //--- ONLY UNCOMMENT ONE OF THESE (depending on which backend server you're running.).... ---//
        //USE THIS LOCAL ONE for local development...
        //const backend_url = "http://localhost:3000"; // URL of the LOCAL backend server (use this if you're running server.js in a separate window!)
        //const backend_url = "http://104.200.25.53/"; //IP address of backend server hosted online, probably don't use this one.
        // TURN THIS ONLINE ONE back on before you run "npm build" and deploy to Vercel!
        const backend_url = "https://snailbunny.site"; // URL of the backend server hosted online!
4. ^ uncomment the localhost:3000 line and comment out the snailbunny.site one. Now your frontend app will send its request to localhost:3000 instead of snailbunny.site.
5. open up 2 terminals -- you should run both the backend server on localhost:3000 with ``node server.js`` (from the /backend folder) and the frontend app on localhost:5173 with ``npm run dev`` from the main directory. Then you'll be able to test changes to both the backend and frontend locally!

### PUSHING YOUR CHANGES
1. switch the localhost:3000 line off, and this one back on: backend_url = https://snailbunny.site";
2. push to the github
3. let shm know you pushed to the github, and they will ssh into the server and pull the changes there
4. the snailbunny.site server will update with your changes!

i will also add the instructions for sshing into the server below, if you are interested, or want to restart/update the server yourself!


**HOW TO LOG IN & RESTART SERVER**

1. `ssh root@104.200.25.53` (this remotely logs you in, as the root user, into the server using its ip address)
2. enter password (ask shm for the password)
3. `screen -r` we’re using a program called screen that virtually leaves a “separate screen on” the server computer even when we close it. there’s not an actual monitor somewhere with the screen on, but that’s the metaphorical idea. this command reattaches to the screen running our backend server
4. you should then see:
    
    ```jsx
    Server running at [http://localhost:3000](http://localhost:3000/)
    Database initialized with artists table
    ```
    
    

this means we’re back on the “screen” that’s running the server

1. you can stop the server with CTRL+C. If the changes you wanted to make are already pushed to github, you can run **`git pull`** now. 
2. start the server again: `node server.js`
3. use **CTRL+A** **CTRL+D** to leave the “screen” on without closing it
4. you can exit safely now!

