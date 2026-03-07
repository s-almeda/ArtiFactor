# Frontend Architecture Reference

## Stack
- **Framework**: React 18 + TypeScript + Vite
- **UI Library**: @xyflow/react (ReactFlow v12) for canvas/node-based interface
- **Routing**: react-router-dom v7
- **Styling**: Tailwind CSS + index.css
- **State**: Context API (5 contexts)
- **Animation**: framer-motion
- **HTTP**: axios

## Entry Point
`main.tsx` → `App.tsx` → Wraps all routes with 5 nested context providers → Renders routes

## Application Architecture

### Provider Hierarchy (outermost to innermost)
```
AppProvider (backend URL, userID, login state, condition)
└─ CanvasProvider (canvas save/load, canvasID, canvasName)
   └─ DnDProvider (drag-and-drop state for nodes)
      └─ ReactFlowProvider (ReactFlow instance)
         └─ NodeProvider (nodes, edges, viewport state)
            └─ PaletteProvider (clipped nodes sidebar)
```

### Routing Structure
```
/ → App.tsx
    ├─ /          → Flow.tsx (main canvas page)
    ├─ /about     → About.tsx
    ├─ /admin     → Admin.tsx
    └─ /test      → TestPage.tsx
```

## Core Files

### `src/App.tsx`
- Defines backend URL (localhost:3000 or snailbunny.site)
- Parses URL params: `?user=X&canvas=Y&con=A|B` (con: control/experimental)
- Wraps app in 6 providers
- Renders: `<TitleBar>` + `<Sidebar>` + `<Palette>` + `<Routes>`
- Handles page unload warnings and ctrl+wheel prevention

### `src/pages/Flow.tsx` (690 lines - main canvas)
- ReactFlow canvas with custom nodes/edges
- Handles node drag, drop, intersection detection
- Auto-saves canvas via debounced NodeContext
- Manages node creation (text/image), deletion, merging
- Implements keyboard shortcuts (Alt+Click for generation)
- Pulls canvas from URL params on mount

### `src/Sidebar.tsx`
- User login/logout UI
- Canvas list (fetch from backend)
- Save/create/delete canvas controls
- Admin panel access

### `src/Palette.tsx`
- Tabbed clipboard for "image" and "text" nodes
- Displays clipped nodes from PaletteContext
- Nodes can be dragged onto canvas

### `src/Toolbar.tsx`
- Canvas controls (likely undo/redo, zoom, etc.)

### `src/TitleBar.tsx`
- Application header/title bar

## Context System

### `context/AppContext.tsx`
**Purpose**: Global app state (user, backend, condition)
**State**: `userID`, `loginStatus`, `backend`, `condition`, `admins[]`
**Methods**: `handleUserLogin()`, `addUser()`
**Usage**: Login flow, backend API URL, experimental condition

### `context/CanvasContext.tsx`
**Purpose**: Canvas persistence (save/load from backend or localStorage)
**State**: `canvasID`, `canvasName`, `lastSaved`
**Methods**: 
- `saveCanvas()` - POST to `/api/save-canvas`
- `pullCanvas()` - GET from `/api/get-canvas/:canvasID`
- `deleteCanvas()` - DELETE `/api/delete-canvas/:userID/:canvasID`
- `createNewCanvas()` - Fetch next ID from backend
- `quickSaveToBrowser()` / `pullCanvasFromBrowser()` - localStorage fallback

### `context/NodeContext.tsx`
**Purpose**: ReactFlow nodes/edges state + auto-save logic
**State**: `nodes[]`, `edges[]`, `currentViewport`
**Methods**:
- `setNodes()`, `setEdges()` - State setters
- `handleOnNodesChange()`, `handleOnEdgesChange()` - ReactFlow change handlers
- `onNodesDelete()`, `deleteNodeById()` - Node deletion
- `drawEdge()` - Connect parent→child nodes
- `canvasToObject()` - Serialize to `ReactFlowJsonObject`
- `saveCurrentViewport()` - Track zoom/pan
- `mergeNodes()` - Combine multiple nodes into one
**Auto-save**: Debounced 1s after changes → calls `saveCanvas()` or `quickSaveToBrowser()`

### `context/DnDContext.tsx`
**Purpose**: Drag-and-drop state for palette→canvas transfers
**State**: `draggableType`, `draggableData`, `dragStartPosition`

### `context/PaletteContext.tsx`
**Purpose**: Clipboard for nodes (text/image)
**State**: `clippedNodes[]`, `activeTab` (image|text)
**Methods**: `addClippedNode()`, `removeNode()`, `loadPalette()`, `getNextPaletteIndex()`

## Node System

### Node Types (`nodes/types.ts`)
All nodes extend `ArtifactorNodeData`:
```typescript
type ArtifactorNodeData = {
  content?: string;
  className?: string;
  xOffset?: number;
  yOffset?: number;
}
```

**Specialized Nodes**:
1. **`ImageWithLookupNode`**
   - Data: `prompt`, `intersections[]`, `similarArtworks[]`, `provenance`, `parentNodeId`, `lookUpOn`
   - Component: `ImageWithLookupNode.tsx` (838 lines)
   - Features: Image display, AI lookup, folder panel for similar artworks

2. **`TextWithKeywordsNode`**
   - Data: `words[]`, `intersections[]`, `similarTexts[]`, `provenance`, `hasNoKeywords`, `hasNoSimilarTexts`, `parentNodeId`
   - Component: `TextWithKeywordsNode.tsx`
   - Features: Editable text, keyword highlighting, text lookup

3. **`LoadingNode`**
   - Placeholder during async operations

**Supporting Types**:
- `Artwork`: Image metadata (image_id, image_urls, artist_names, descriptions, relatedKeywords, rights, distance)
- `Keyword`: Word with metadata (entryId, databaseValue, images[], isArtist, type, aliases[], descriptions[], relatedKeywords)
- `Word`: Simple `{value: string}`

### Node Registration (`nodes/index.ts`)
```typescript
export const nodeTypes = {
  default: LoadingNode,
  image: ImageWithLookupNode,
  text: TextWithKeywordsNode,
  imagewithlookup: ImageWithLookupNode,
  textwithkeywords: TextWithKeywordsNode,
};
```

## Edge System

### `edges/WireEdge.tsx`
- Custom Bezier path edge
- Midpoint delete button
- Registered in `edges/index.ts`

## Components

### `components/NavigationButtons.tsx`
- Navigation controls for nodes (likely prev/next for similar artworks)

### `components/DynamicDescription.tsx`
- Dynamic description display for artwork/keyword metadata

## Utilities

### `utils/utilityFunctions.ts`
- `stringToWords(str)` → `Word[]`
- `wordsToString(words)` → `string`
- `keywordJSONtoKeyword(json)` → `Keyword` - Parse backend JSON to typed Keyword
- `calcNearbyPosition()` - Calculate offset positions for new nodes

### `utils/commonComponents.tsx`
- Shared React components

## Hooks

### `hooks/useClipboard.ts`
- Custom clipboard logic for nodes

## Backend Integration

**Base URL**: Defined in `App.tsx` as `backend_url`

**API Routes Used**:
- `POST /api/check-for-keywords` - Text→keyword matching
- `POST /api/get-similar-texts` - Text vector search
- `POST /api/get-similar-images` - Image vector search
- `POST /api/get-artworks-by-ids` - Batch artwork fetch
- `POST /api/generate-text` - Image→text (CLIP interrogator via Replicate)
- `POST /api/generate-image` - Text→image (Stable Diffusion via Reagent)
- `POST /api/save-canvas` - Persist canvas to DB
- `GET /api/get-canvas/:canvasID` - Load canvas from DB
- `DELETE /api/delete-canvas/:userID/:canvasID` - Delete canvas
- `GET /api/list-canvases/:userID` - List user's canvases
- `GET /api/next-canvas-id/:userID` - Get next available canvas ID
- `POST /api/add-user` - Register new user
- `GET /api/list-users` - List all users

## URL Parameters
- `?user=<userID>` - Auto-login user
- `?canvas=<canvasID>` - Load specific canvas
- `?con=A|B` - A=control (no lookup), B=experimental (lookup enabled)

## Data Flow Patterns

### Canvas Load
1. `Flow.tsx` reads URL params
2. Calls `CanvasContext.pullCanvas(canvasID)`
3. Fetches from backend → parses JSON
4. Sets `NodeContext.setNodes()`, `setEdges()`, viewport
5. ReactFlow renders

### Canvas Save
1. User edits nodes/edges
2. `NodeContext` debounces changes (1s)
3. Calls `CanvasContext.saveCanvas(canvasToObject())`
4. POST to backend with `{userID, canvasID, canvasName, canvasJSONObject, timestamp}`

### Node Creation (Drag-from-Palette)
1. User drags from `Palette.tsx`
2. `DnDContext` stores `draggableType` + `draggableData`
3. Drop on canvas triggers `Flow.tsx` drop handler
4. Creates node via `NodeContext.setNodes()`
5. Auto-save triggers

### Node Lookup (Alt+Click)
1. User Alt+Clicks node
2. `ImageWithLookupNode` or `TextWithKeywordsNode` triggers lookup
3. Sends request to backend (`/api/get-similar-images` or `/api/get-similar-texts`)
4. Updates node data with `similarArtworks[]` or `similarTexts[]`
5. Renders folder panel with results

### Node Intersection
1. User drags node over another
2. `Flow.tsx` `onNodeDrag` → `getIntersectingNodes()`
3. Updates both nodes' `intersections[]` array
4. Nodes re-render with intersection highlights

## Public Assets
`public/` folder is empty (static assets served from CDN or backend)

## Configuration Files
- `tailwind.config.js` - Tailwind CSS config
- `postcss.config.js` - PostCSS config
- `vite.config.ts` - Vite bundler config
- `tsconfig.json` / `tsconfig.node.json` - TypeScript configs
- `vercel.json` - Vercel deployment config

## Key Behaviors

### Auto-Save
- **Trigger**: Any node/edge/viewport change
- **Debounce**: 1 second
- **Logic**: If logged in → backend, else → localStorage

### Offline Mode
- Canvas saves to `localStorage` when not logged in
- Key format: `localStorage[canvasID]` = `{nodes, edges, viewport}`
- Canvas name: `localStorage[canvasID-name]` = string

### Node Provenance
- `"history"` - From database (artwork/keyword lookup)
- `"user"` - User-created or edited
- `"ai"` - AI-generated (image generation, text interrogation)

### Intersection Detection
- Uses ReactFlow's `getIntersectingNodes()`
- Stores intersecting node IDs in each node's `intersections[]`
- Used for synthesis/merging behaviors

## Critical Implementation Notes

1. **State Management**: All canvas state lives in `NodeContext`, not local state in `Flow.tsx`
2. **Type Safety**: Nodes are strongly typed via `AppNode<T extends ArtifactorNodeData>`
3. **Serialization**: Canvas JSON = `{nodes, edges, viewport}` (ReactFlowJsonObject)
4. **User Flow**: URL params → Context state → Component props (one-way data flow)
5. **Canvas Identity**: `canvasID` format = `{userID}-{number}` (e.g., "alice-0", "alice-1")
6. **Lookup Toggle**: Controlled by `condition` prop ("control" disables, "experimental" enables)
7. **Admin Access**: Hardcoded admin list in `AppContext` (shm, elaine, ethan, sophia, bob)

## File Count Summary
- **Total**: 31 frontend files
- **Contexts**: 5 files
- **Nodes**: 5 files (3 components + types + index)
- **Pages**: 4 files
- **Components**: 2 files
- **Utils**: 2 files
- **Hooks**: 1 file
- **Edges**: 2 files
- **Root**: 7 files (App, main, Sidebar, Palette, TitleBar, Toolbar, index.css)

---

## Backend Architecture

### Server Stack (`BackendServer/`)
- **Runtime**: Node.js (Express.js)
- **Database**: SQLite3 via `better-sqlite3` + `sqlite` wrapper
- **HTTP Client**: axios (for proxying to external APIs)
- **Port**: 3000
- **CORS**: `https://arti-factor.vercel.app`, `http://localhost:5173`
- **Payload Limit**: 50MB (for large canvas JSON with images)

### Files
- **`server.js`**: Main Express server (871 lines) - routes, logging, API proxying
- **`database.js`**: SQLite schema initialization + admin seeding
- **`default_canvas_data.json`**: Template for new canvases `{nodes: [], edges: [], viewport: {x:0, y:0, zoom:1}}`
- **`log-*.txt`**: Timestamped request logs (all API calls + responses)

### Database Schema

#### `users` Table
```sql
CREATE TABLE users (
  userId TEXT PRIMARY KEY,
  clippings TEXT NOT NULL DEFAULT '[]'  -- JSON array of palette nodes
)
```
- **Purpose**: User accounts + clipboard/palette state
- **No passwords**: Auth via URL params only

#### `canvases` Table
```sql
CREATE TABLE canvases (
  canvasId TEXT PRIMARY KEY,           -- Format: {userId}-{number}
  userId TEXT NOT NULL,
  canvasName TEXT NOT NULL,
  FOREIGN KEY (userId) REFERENCES users(userId) ON DELETE CASCADE
)
```
- **Purpose**: Canvas metadata (name, owner)
- **canvasId Format**: `alice-0`, `alice-1`, `bob-0` etc.
- **No JSON blob here**: Actual canvas data stored in `versions` table

#### `versions` Table
```sql
CREATE TABLE versions (
  versionId TEXT PRIMARY KEY,          -- Format: {canvasId}-{ISO_TIMESTAMP}
  canvasId TEXT NOT NULL,
  timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  jsonBlob TEXT NOT NULL,              -- Serialized ReactFlowJsonObject
  FOREIGN KEY (canvasId) REFERENCES canvases(canvasId) ON DELETE CASCADE
)
```
- **Purpose**: Version history (multiple snapshots per canvas)
- **jsonBlob**: Stringified `{nodes: [], edges: [], viewport: {}}`
- **Versioning Logic**: New version created only if node IDs changed, else overwrite latest

### Canvas JSON Structure

#### Stored in `versions.jsonBlob`
```json
{
  "nodes": [
    {
      "id": "node-uuid-123",
      "type": "image" | "text" | "imagewithlookup" | "textwithkeywords",
      "position": {"x": 100, "y": 200},
      "data": {
        "content": "text content or image URL",
        "prompt": "AI prompt or artwork title",
        "words": [{"value": "word1"}, {"value": "word2"}],
        "similarArtworks": [...],
        "similarTexts": [...],
        "intersections": [{"id": "node-id", "position": {...}, "content": "..."}],
        "provenance": "history" | "user" | "ai",
        "parentNodeId": "parent-uuid",
        "lookUpOn": true,
        "hasNoKeywords": false,
        "className": "...",
        "xOffset": 10,
        "yOffset": 20
      }
    }
  ],
  "edges": [
    {
      "id": "edge-uuid-456",
      "source": "node-uuid-123",
      "target": "node-uuid-789",
      "type": "wire"
    }
  ],
  "viewport": {"x": 0, "y": 0, "zoom": 1}
}
```

### API Endpoints

#### User Management
- **`POST /api/add-user`**: Create user → `INSERT INTO users (userId) VALUES (?)`
- **`GET /api/list-users`**: List all users + their canvases (admin only)

#### Canvas CRUD
- **`POST /api/save-canvas`**: 
  - Input: `{userID, canvasID, canvasName, canvasJSONObject, timestamp}`
  - Logic:
    1. `INSERT OR UPDATE canvases` (canvasName only)
    2. Compare node IDs with latest version
    3. If changed: `INSERT INTO versions (...)` new row
    4. Else: `UPDATE versions SET jsonBlob=...` latest row
  - Response: `{success: true}`

- **`GET /api/get-canvas/:canvasID`**:
  - Logic:
    1. `SELECT canvasName FROM canvases WHERE canvasId=?`
    2. `SELECT * FROM versions WHERE canvasId=? ORDER BY timestamp DESC LIMIT 1`
    3. Parse `jsonBlob` → extract `{nodes, edges, viewport}`
  - Response: `{success: true, canvas: {canvasID, canvasName, nodes, edges, viewport}, timestamp}`

- **`DELETE /api/delete-canvas/:userID/:canvasID`**:
  - Logic:
    1. `DELETE FROM canvases WHERE canvasId=? AND userId=?`
    2. Cascade deletes all versions (via FOREIGN KEY)
  - Response: `{success: true, message: "..."}`

- **`GET /api/list-canvases/:userID`**:
  - Query: `SELECT canvasId, canvasName FROM canvases WHERE userId=?`
  - Response: `{success: true, canvases: [{canvasId, canvasName}...]}`

- **`GET /api/next-canvas-id/:userID`**:
  - Query: `SELECT MAX(CAST(SUBSTR(canvasId, INSTR(canvasId, '-') + 1) AS INTEGER)) FROM canvases WHERE userId=?`
  - Returns: `{userId}-{maxNumber+1}`

#### ML/AI Proxied Routes (Flask Server)
**Flask URL**: `https://data.snailbunny.site` or `http://localhost:8080`

- **`POST /api/check-for-keywords`** → Flask `/keyword_check`
  - Input: `{text, threshold}`
  - Output: Keyword matches with confidence scores

- **`POST /api/get-similar-texts`** → Flask `/lookup_text`
  - Input: `{query, top_k}`
  - Output: `[{entryId, value, distance, ...}]`

- **`POST /api/get-similar-images`** → Flask `/image`
  - Input: `{image}` (base64 or URL)
  - Output: `[{image_id, image_url, artist_names, distance, ...}]`

- **`POST /api/get-artworks-by-ids`** → Flask `/lookup_entry` (batched)
  - Input: `{entryIds: [id1, id2, ...]}`
  - Output: `{artworks: [{image_id, image_urls, artist_names, descriptions, relatedKeywords, ...}]}`

#### External AI APIs
- **`POST /api/generate-text`**: Image→text via Replicate CLIP Interrogator
  - Requires: `REPLICATE_API_TOKEN` env var
  - Model: `pharmapsychotic/clip-interrogator:8151e1c9...`
  - Returns: `{text: "keyword1, keyword2, ..."}`

- **`POST /api/generate-image`**: Text→image via Reagent API
  - Model: Black Forest Labs Schnell
  - Endpoint: `https://noggin.rea.gent/peaceful-spoonbill-8088`
  - Returns: `{imageUrl: "https://..."}`

#### Utility Routes
- **`GET /`**: Health check → `"Hello World!"`
- **`GET /health_check`**: Test all Flask endpoints → HTML report
- **`GET /overview`**: Display current log file
- **`GET /fresh`**: Rotate to new log file

### Data Persistence Patterns

#### Canvas Save Flow
1. Frontend: `NodeContext` debounces changes (1s) → calls `CanvasContext.saveCanvas()`
2. `POST /api/save-canvas` with `{userID, canvasID, canvasName, canvasJSONObject, timestamp}`
3. Backend:
   - Check if `canvasId` exists in `canvases` table
   - If exists: `UPDATE canvases SET canvasName=?`
   - If new: `INSERT INTO canvases (...)`
4. Backend queries latest version: `SELECT * FROM versions WHERE canvasId=? ORDER BY timestamp DESC LIMIT 1`
5. Compare `latestVersion.jsonBlob.nodes[].id` vs `newCanvas.nodes[].id`
6. If node IDs changed: Insert new row → `INSERT INTO versions (versionId, canvasId, timestamp, jsonBlob)`
7. Else: Update existing row → `UPDATE versions SET jsonBlob=?, timestamp=?`

#### Canvas Load Flow
1. Frontend: `Flow.tsx` reads URL param `?canvas=alice-0`
2. `GET /api/get-canvas/alice-0`
3. Backend:
   - `SELECT canvasName FROM canvases WHERE canvasId='alice-0'`
   - `SELECT * FROM versions WHERE canvasId='alice-0' ORDER BY timestamp DESC LIMIT 1`
   - Parse `jsonBlob` string → JSON object
4. Response: `{canvas: {canvasID, canvasName, nodes, edges, viewport}, timestamp}`
5. Frontend: `NodeContext.setNodes(nodes)`, `setEdges(edges)`, `setViewport(viewport)`

#### Versioning Strategy
- **Version created**: When node IDs array changes (nodes added/removed)
- **Version updated**: When only node data/positions change (same IDs)
- **versionId format**: `{canvasId}-{ISO_timestamp}` e.g., `alice-0-2026-03-06T12:34:56.789Z`
- **Retrieval**: Always fetch latest version (`ORDER BY timestamp DESC LIMIT 1`)

### Logging System
- **Middleware**: Logs every request → `[timestamp] method URL | Body: {...} | Query: {...}`
- **Log file**: `log-{ISO_timestamp}.txt` (new file per server restart)
- **Logged events**: API calls, ML responses, image generation, errors
- **Rotation**: Via `GET /fresh` endpoint

### External Service Integration

#### Flask ML Server (Python/Docker)
- **Purpose**: Vector search (CLIP embeddings), keyword extraction
- **Location**: `KnowledgeServer/` (separate Docker container)
- **Database**: Art history knowledge base with ~8000 artworks + keywords
- **Models**: CLIP (image+text embeddings), UMAP (dimensionality reduction)

#### Replicate
- **Purpose**: AI image-to-text (CLIP Interrogator)
- **Auth**: API token in `.env` → `process.env.REPLICATE_API_TOKEN`
- **Fallback**: Returns 503 if token missing

#### Reagent
- **Purpose**: AI text-to-image (Stable Diffusion)
- **Auth**: Hardcoded bearer token
- **Format**: WebP image returned as redirect URL

### Performance Optimizations
- **Debounced saves**: 1s delay prevents excessive DB writes
- **Batch fetches**: `get-artworks-by-ids` uses `Promise.all()` for parallel requests
- **Version deduplication**: Only save new version if structure changed
- **50MB payload limit**: Supports large canvases with embedded images
- **SQLite async**: All queries use `await` with connection pooling

### Error Handling
- Missing fields → `400 Bad Request`
- User/canvas not found → `404 Not Found`
- ML server timeout → `500 Internal Server Error`
- Missing API token → `503 Service Unavailable`
- All errors auto-logged to file
