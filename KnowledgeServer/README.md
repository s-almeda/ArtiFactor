# **Making Requests to the Map Generation API**

### Submitting a Map Generation Job

Endpoint: /submit_map_job
Method: POST
Content-Type: application/json

***Request Body Structure:***

```python
{
  "debug": true,
  "numKeywords": 50,
  "weights": {
    "clip": 0.3,
    "resnet": 0.5,
    "keyword_semantic": 0.2,
    "keyword_bias": 0.6,
    "debug": true
  },
  "umap": {
    "min_dist": 0.9,
    "parallel": true,
    "random_state": 42  // Only included if parallel is false
  },
  "compression": {
    "threshold_percentile": 90,
    "compression_factor": 0.3
  },
  "padding_factor": 0.1,
  "n_clusters": 10  // Optional, will auto-determine if omitted
}
```

### Job Status Polling

1. Submit the job to receive a `job_id` in the response
2. Poll for status at `/job_status/{job_id}` (recommended interval: 2 seconds)
3. Check the `status` field in the response:
    - `"completed"`: Job is done, retrieve results using the `cache_key`
    - `"failed"`: Check the `error` field for failure details
    - Any other status: Continue polling

### Getting Results

**Endpoint:** `/get_result/{cache_key}`

**Method:** `GET`

When the job status returns `completed`, use the `cache_key` from the status response to fetch results.

### Sample JavaScript Frontend Code

```jsx
// 1. Submit job
const response = await fetch('/submit_map_job', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify(params)
});
const jobData = await response.json();

// If result is cached, it's returned immediately
if (jobData.result) {
  console.log('Cached result:', jobData.result);
  // Process the result
  return;
}

// 2. Poll for status
const jobId = jobData.job_id;
let completed = false;
while (!completed) {
  const statusResponse = await fetch(`/job_status/${jobId}`);
  const statusData = await statusResponse.json();
  
  if (statusData.status === 'completed') {
    // 3. Get result
    const resultResponse = await fetch(`/get_result/${statusData.cache_key}`);
    const result = await resultResponse.json();
    console.log('Final result:', result);
    completed = true;
  } else if (statusData.status === 'failed') {
    console.error('Job failed:', statusData.error);
    completed = true;
  } else {
    // Still processing, wait and poll again
    console.log('Status:', statusData.message);
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
}
```

The result will contain a hierarchical structure with multiple map levels, artworks data, and cluster information that can be used to visualize the map.

# V3 RESPONSE STRUCTURE

```python
{
  // === METADATA SECTION ===
  "artworks": {
    [image_id: string]: {
      "title": string,                   // Artwork title, e.g. "The Kiss"
      "artist": string,                 // Primary artist (first of artist_names)
      "artist_names": string[],        // All known artist names
      "image_urls": {                  // All available sizes and formats (parsed from JSON)
        [size_label: string]: string   // e.g., "large": "https://..."
      },
      "thumbnail_url": string,         // Best available small image (for previews)
      "url": string,                   // Best available large image (for display)
      "descriptions": {               // Raw, unflattened descriptions grouped by source
        [source: string]: {
          [field: string]: any        // Arbitrary fields per source (e.g., "medium", "description", "category")
        }
      },
      "rights": string,                // Rights info or citation (e.g., Public Domain)
      "keywords": string[]             // Up to 10 keywords per artwork (curated or derived)
    }
  },

  // === CLUSTERING + SPATIAL ORGANIZATION SECTION ===
  // These are hierarchical Voronoi maps

  "level_1": [                         // Base layer: fine-grained Voronoi clusters
    {
      "cluster_id": number,           // Unique ID for this cluster (within level)
      "cluster_label": string,        // Optional label based on shared content (e.g. artist name)
      "centroid": { x: float, y: float }, // Center of the Voronoi cell (normalized 0-1)
      "voronoi_vertices": [           // Polygon defining the region for this cluster (in 2D view space)
        [x: float, y: float], ...
      ],
      "representative_artworks": string[], // Image IDs representative of this cluster
      "artworks_map": [               // Artworks contained in this cluster
        {
          "id": string,               // Display instance ID (often prefixed with "w_")
          "coords": { x: float, y: float } // Position in 2D space (normalized)
        },
        ...
      ]
    },
    ...
  ],

  "level_2": [                         // Merged from level_1 clusters
    {
      "cluster_id": number,
      "cluster_label": string,
      "centroid": { x, y },
      "voronoi_vertices": [[x, y], ...],
      "representative_artworks": [string, string, ...],
      "child_clusters": [number, number, ...] // cluster_ids from level_1 this cluster merges
    },
    ...
  ],

  "level_3": [                         // Final merged clusters (e.g., at most 4 regions)
    {
      "cluster_id": number,
      "cluster_label": string,
      "centroid": { x, y },
      "voronoi_vertices": [[x, y], ...],
      "representative_artworks": [string, ...],
      "child_clusters": [number, number, ...] 
    },
    ...
  ],

  // === SYSTEM METADATA ===
  "cache_key": string,                 // Unique hash identifying this specific result
  "cached": boolean,                  // Whether this result was loaded from cache
  "success": true                      // Always true if data loaded successfully
}

```

truncated example of a response:

<aside>
💡

1. **artworks**: {4eaefc4976e78f0001009e86: {…}, 4eaefdd86899c800010081e1: {…}, 4eb0654269c04b00010096da: {…}, 5033edb6946de10002000150: {…}, 506dfe925169220002000748: {…}, …}
    1. **4d8b93b04eb68a1b2c001b9d**:
        1. **artist**: "Édouard Manet"
        2. **artist_names**: ['Édouard Manet']
        3. **descriptions**:
            1. **artsy**:
                1. **additional_information**: "[Image source](http://commons.wikimedia.org/wiki/File:Edouard_Manet_-_Luncheon_on_the_Grass_-_Google_Art_Project.jpg)"
                2. **category**: "Painting"
                3. **collecting_institution**: "Musée d'Orsay, Paris"
                4. **date**: "1863"
                5. **description**: ""
                6. **medium**: "Oil on canvas"
                7. [[Prototype]]: Object
            2. [[Prototype]]: Object
        4. **image_urls**: {large: 'https://d32dm0rphc51dk.cloudfront.net/zFA7cwdkWxbIrmuAAd21VA/large.jpg', large_rectangle: 'https://d32dm0rphc51dk.cloudfront.net/zFA7cwdkWxbIrmuAAd21VA/large_rectangle.jpg', larger: 'https://d32dm0rphc51dk.cloudfront.net/zFA7cwdkWxbIrmuAAd21VA/larger.jpg', medium: 'https://d32dm0rphc51dk.cloudfront.net/zFA7cwdkWxbIrmuAAd21VA/medium.jpg', medium_rectangle: 'https://d32dm0rphc51dk.cloudfront.net/zFA7cwdkWxbIrmuAAd21VA/medium_rectangle.jpg', …}
        5. **keywords**: (10) ['Édouard Manet', '1860–1969', '19th Century', 'Cultural Commentary', 'Dark Colors', 'Dense Composition', 'Eye Contact', 'Figurative Art', 'Figures in Nature', 'Flatness']
        6. **rights**: "Source: Wikimedia Commons / Public Domain"
        7. **thumbnail_url**: "https://d32dm0rphc51dk.cloudfront.net/zFA7cwdkWxbIrmuAAd21VA/small.jpg"
        8. **title**: "Luncheon on the Grass (Le Déjeuner sur l'herbe)"
        9. **url**: "https://d32dm0rphc51dk.cloudfront.net/zFA7cwdkWxbIrmuAAd21VA/large.jpg"
        10. [[Prototype]]: Object
2. **cache_key**: "map_v3_8761d0ff5c92"
3. **cached**: false
4. **level_1**: Array(15)
    1. **0**:
        1. **artworks_map**: Array(299)
            1. **0**:
                1. **coords**: {x: 0.4184559114069035, y: 0.023876945633154782}
                2. **id**: "w_6f8c2f30a5d64bffb45114d"
                3. [[Prototype]]: Object
            2. …
5. **level_2**: Array(5)
    1. **0**:
        1. **centroid**: {x: 0.6758781487542728, y: 0.8683435481931429}
        2. **child_clusters**: Array(2)
            1. **0**: 3
            2. **1**: 11
            3. **length**: 2
            4. [[Prototype]]: Array(0)
        3. **cluster_id**: 19
        4. **cluster_label**: "Aleksey Savrasov & Henri de Toulouse-Lautrec"
        5. **cluster_info**: {                          // NEW FIELD
            1. **keywords**: [
                1. **term**: "impressionism"
                2. **count**: 45
                3. **percentage**: 75.0
                4. **term**: "landscape"
                5. **count**: 38
                6. **percentage**: 63.3
            2. **mediums**: [
                1. **term**: "oil on canvas"
                2. **count**: 42
                3. **percentage**: 70.0
            3. **artists**: [
                1. **name**: "Aleksey Savrasov"
                2. **count**: 35
                3. **percentage**: 58.3
                4. **name**: "Henri de Toulouse-Lautrec"
                5. **count**: 25
                6. **percentage**: 41.7
            4. **date_range**: {
                1. **min_year**: 1865
                2. **max_year**: 1899
                3. **formatted**: "1865-1899"
                4. **count**: 52
            }
        6. **representative_artworks**: (3) ['w_16457b729c7d4a4197e33fb', 'w_3cd89ba4cd8843899f928ab', 'w_0ebf2cfa53a342b195b3716']
        7. **voronoi_vertices**: (9) [Array(2), Array(2), Array(2), Array(2), Array(2), Array(2), Array(2), Array(2), Array(2)]
            1. **0**: Array(2)
                1. **0**: 0.5497507237451149
                2. **1**: 0.6859427594549662
                3. **length**: 2
    2. **1**: {centroid: {…}, child_clusters: Array(2), cluster_id: 20, cluster_label: 'Charles M. Russell & Raphael', representative_artworks: Array(3), …}
    3. **2**: {centroid: {…}, child_clusters: Array(2), cluster_id: 22, cluster_label: 'Alfred Sisley', representative_artworks: Array(3), …}
    4. **3**: {centroid: {…}, child_clusters: Array(2), cluster_id: 23, cluster_label: 'Cluster 24 (740 artworks)', representative_artworks: Array(3), …}
    5. **4**: {centroid: {…}, child_clusters: Array(2), cluster_id: 24, cluster_label: 'Albrecht Durer & Albert Bierstadt', representative_artworks: Array(3), …}
    6. **length**: 5
6. **level_3**: Array(3)
    1. **0**:
        1. **centroid**: {x: 0.20277211204090126, y: 0.3177451238403868}
        2. **child_clusters**: (2) [16, 21]
        3. **cluster_id**: 22
        4. **cluster_label**: "Alfred Sisley"
        5. **representative_artworks**: (3) ['w_75f067f0363f40c28b1e61f', 'w_062d5685dcd84c90a452520', 'w_3297a124f66b42bb92546ca']
        6. **voronoi_vertices**: (13) [Array(2), Array(2), Array(2), Array(2), Array(2), Array(2), Array(2), Array(2), Array(2), Array(2), Array(2), Array(2), Array(2)]
        7. [[Prototype]]: Object
    2. **1**: {centroid: {…}, child_clusters: Array(2), cluster_id: 25, cluster_label: 'Albrecht Durer', representative_artworks: Array(3), …}
    3. **2**: {centroid: {…}, child_clusters: Array(2), cluster_id: 26, cluster_label: 'Cluster 27 (1107 artworks)', representative_artworks: Array(3), …}
    4. **length**: 3
7. **success**: true
</aside>

### Cluster Label Generation

Cluster labels are intelligently generated based on the artworks they contain:

- **Smart labeling**: Uses top shared keywords, mediums, artists, and date ranges
- **Fallback strategy**: If <50% of artworks share keywords/mediums, falls back to top artists
- **Examples**:
  - Rich cluster: `"baroque, classical figure, sketch, lithograph, marcel duchamp, 1850-1900"`
  - Diverse cluster: `"duchamp, monet, van gogh, 1850-1920"`
  - Mixed periods: `"contemporary art, picasso, basquiat, warhol"`

The `cluster_info` field provides the detailed analytics used to generate these labels, including exact counts and percentages for frontend display.