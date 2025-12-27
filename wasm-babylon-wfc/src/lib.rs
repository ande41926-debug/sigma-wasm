use wasm_bindgen::prelude::*;
use std::sync::{LazyLock, Mutex};

/// Tile type enumeration for the 11 different tile types
/// 
/// **Learning Point**: This enum represents all possible tile types in the WFC grid.
/// Each tile type has specific edge compatibility rules that determine which tiles
/// can be adjacent to each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum TileType {
    Grass = 0,
    Floor = 1,
    WallNorth = 2,
    WallSouth = 3,
    WallEast = 4,
    WallWest = 5,
    CornerNE = 6,
    CornerNW = 7,
    CornerSE = 8,
    CornerSW = 9,
    Door = 10,
}

/// Edge type enumeration for tile adjacency rules
/// 
/// **Learning Point**: Each tile has 4 edges (North, South, East, West).
/// The WFC algorithm uses edge compatibility to determine which tiles can
/// be placed next to each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum EdgeType {
    Empty = 0,
    Wall = 1,
    Floor = 2,
    Grass = 3,
    Door = 4,
}

/// Edge compatibility structure for a tile type
/// 
/// **Learning Point**: This defines what edge types each tile type has on its
/// four sides. The WFC algorithm uses this to ensure tiles are placed correctly.
struct TileEdges {
    north: EdgeType,
    south: EdgeType,
    east: EdgeType,
    west: EdgeType,
}

impl TileEdges {
    fn new(north: EdgeType, south: EdgeType, east: EdgeType, west: EdgeType) -> Self {
        TileEdges {
            north,
            south,
            east,
            west,
        }
    }
}

/// Get edge compatibility for a tile type
/// 
/// **Learning Point**: This function defines the edge types for each of the 11 tile types.
/// When implementing WFC, tiles can only be adjacent if their edges match (e.g., 
/// a tile's north edge must match the neighbor's south edge).
fn get_tile_edges(tile_type: TileType) -> TileEdges {
    match tile_type {
        TileType::Grass => TileEdges::new(EdgeType::Grass, EdgeType::Grass, EdgeType::Grass, EdgeType::Grass),
        TileType::Floor => TileEdges::new(EdgeType::Floor, EdgeType::Floor, EdgeType::Floor, EdgeType::Floor),
        TileType::WallNorth => TileEdges::new(EdgeType::Empty, EdgeType::Wall, EdgeType::Wall, EdgeType::Wall),
        TileType::WallSouth => TileEdges::new(EdgeType::Wall, EdgeType::Empty, EdgeType::Wall, EdgeType::Wall),
        TileType::WallEast => TileEdges::new(EdgeType::Wall, EdgeType::Wall, EdgeType::Empty, EdgeType::Wall),
        TileType::WallWest => TileEdges::new(EdgeType::Wall, EdgeType::Wall, EdgeType::Wall, EdgeType::Empty),
        TileType::CornerNE => TileEdges::new(EdgeType::Empty, EdgeType::Wall, EdgeType::Empty, EdgeType::Wall),
        TileType::CornerNW => TileEdges::new(EdgeType::Empty, EdgeType::Wall, EdgeType::Wall, EdgeType::Empty),
        TileType::CornerSE => TileEdges::new(EdgeType::Wall, EdgeType::Empty, EdgeType::Empty, EdgeType::Wall),
        TileType::CornerSW => TileEdges::new(EdgeType::Wall, EdgeType::Empty, EdgeType::Wall, EdgeType::Empty),
        TileType::Door => TileEdges::new(EdgeType::Door, EdgeType::Door, EdgeType::Door, EdgeType::Door),
    }
}

/// Check if two edge types are compatible
/// 
/// **Learning Point**: For WFC to work, edges must match. This function determines
/// if two edge types can be adjacent (e.g., Wall matches Wall, Empty matches Empty).
fn edges_compatible(edge1: EdgeType, edge2: EdgeType) -> bool {
    edge1 == edge2
}

/// Check if a tile type can be placed at a position given its neighbors
/// 
/// **Learning Point**: This is the core constraint checking for WFC. Before placing
/// a tile, we check if its edges are compatible with all existing neighbors.

/// Voronoi seed point for grass generation
/// 
/// **Learning Point**: Voronoi diagrams divide space into regions based on seed points.
/// Each cell belongs to the region of its closest seed point.
struct VoronoiSeed {
    x: f64,
    y: f64,
    is_grass: bool,
}

/// Generate Voronoi diagram and mark grass regions
/// 
/// **Learning Point**: This function creates natural-looking grass patches using Voronoi noise.
/// It generates random seed points, then determines which cells belong to grass regions
/// based on distance to seeds. This prevents large uniform quadrants of grass.
fn generate_voronoi_grass(
    width: i32,
    height: i32,
    num_seeds: usize,
) -> [[bool; 50]; 50] {
    let mut grass_map = [[false; 50]; 50];
    
    // Generate random seed points
    let mut seeds: Vec<VoronoiSeed> = Vec::new();
    for _ in 0..num_seeds {
        let x = js_random() * width as f64;
        let y = js_random() * height as f64;
        // Randomly decide if this seed region should be grass (about 40% grass)
        let is_grass = js_random() < 0.4;
        seeds.push(VoronoiSeed { x, y, is_grass });
    }
    
    // For each cell, find closest seed and mark as grass if seed is grass
    for y in 0..height {
        for x in 0..width {
            let mut min_dist = f64::MAX;
            let mut closest_is_grass = false;
            
            for seed in &seeds {
                let dx = x as f64 - seed.x;
                let dy = y as f64 - seed.y;
                let dist = dx * dx + dy * dy; // Squared distance (no need to sqrt)
                
                if dist < min_dist {
                    min_dist = dist;
                    closest_is_grass = seed.is_grass;
                }
            }
            
            grass_map[y as usize][x as usize] = closest_is_grass;
        }
    }
    
    grass_map
}

/// Wave function (superposition) for a single cell
/// 
/// **Learning Point**: In WFC, each cell maintains a "wave function" - a set of all
/// possible tile types that could be placed there. The entropy is the number of
/// possibilities. Cells with lower entropy are collapsed first.
struct WaveCell {
    possible_tiles: Vec<TileType>,
}

impl WaveCell {
    fn new() -> Self {
        // Start with all tile types possible
        WaveCell {
            possible_tiles: vec![
                TileType::Grass,
                TileType::Floor,
                TileType::WallNorth,
                TileType::WallSouth,
                TileType::WallEast,
                TileType::WallWest,
                TileType::CornerNE,
                TileType::CornerNW,
                TileType::CornerSE,
                TileType::CornerSW,
                TileType::Door,
            ],
        }
    }
    
    fn entropy(&self) -> usize {
        self.possible_tiles.len()
    }
    
    fn collapse(&mut self) -> Option<TileType> {
        if self.possible_tiles.is_empty() {
            return None;
        }
        
        // Randomly select one of the possible tiles
        let index = (js_random() * self.possible_tiles.len() as f64) as usize;
        let tile = self.possible_tiles[index];
        self.possible_tiles = vec![tile];
        Some(tile)
    }
    
    fn remove_tile(&mut self, tile: TileType) {
        self.possible_tiles.retain(|&t| t != tile);
    }
}

/// WFC state structure with wave function
/// 
/// **Learning Point**: This follows the same state management pattern as other WASM modules.
/// We use LazyLock<Mutex<State>> to manage global mutable state safely.
struct WfcState {
    grid: [[Option<TileType>; 50]; 50],
    wave: [[WaveCell; 50]; 50],
    pre_constraints: [[Option<TileType>; 50]; 50], // Pre-constraints set before WFC
    width: i32,
    height: i32,
}

impl WfcState {
    fn new() -> Self {
        // Initialize wave array element by element (can't use array literal with non-Copy types)
        // Use MaybeUninit for safe initialization
        let mut wave: [[std::mem::MaybeUninit<WaveCell>; 50]; 50] = unsafe {
            std::mem::MaybeUninit::uninit().assume_init()
        };
        
        // Initialize each wave cell
        for y in 0..50 {
            for x in 0..50 {
                wave[y][x].write(WaveCell::new());
            }
        }
        
        // Safe to assume_init because all elements are initialized
        let wave: [[WaveCell; 50]; 50] = unsafe {
            std::mem::transmute(wave)
        };
        
        WfcState {
            grid: [[None; 50]; 50],
            wave,
            pre_constraints: [[None; 50]; 50],
            width: 50,
            height: 50,
        }
    }
    
    fn clear(&mut self) {
        self.grid = [[None; 50]; 50];
        self.pre_constraints = [[None; 50]; 50];
        
        // Reinitialize each wave cell
        for y in 0..50 {
            for x in 0..50 {
                self.wave[y][x] = WaveCell::new();
            }
        }
    }
    
    /// Set a pre-constraint at a specific position
    /// Returns true if the constraint was set successfully
    fn set_pre_constraint(&mut self, x: i32, y: i32, tile_type: TileType) -> bool {
        if x >= 0 && x < self.width && y >= 0 && y < self.height {
            self.pre_constraints[y as usize][x as usize] = Some(tile_type);
            true
        } else {
            false
        }
    }
    
    /// Clear all pre-constraints
    fn clear_pre_constraints(&mut self) {
        self.pre_constraints = [[None; 50]; 50];
    }
    
    fn get_tile(&self, x: i32, y: i32) -> Option<TileType> {
        if x >= 0 && x < self.width && y >= 0 && y < self.height {
            self.grid[y as usize][x as usize]
        } else {
            None
        }
    }
    
    /// Find the cell with lowest entropy (fewest possibilities)
    /// 
    /// **Learning Point**: WFC always collapses the cell with lowest entropy first.
    /// This minimizes contradictions and ensures the algorithm progresses efficiently.
    fn find_lowest_entropy(&self) -> Option<(i32, i32)> {
        let mut min_entropy = usize::MAX;
        let mut best_pos: Option<(i32, i32)> = None;
        
        for y in 0..self.height {
            for x in 0..self.width {
                if self.grid[y as usize][x as usize].is_some() {
                    continue; // Already collapsed
                }
                
                let entropy = self.wave[y as usize][x as usize].entropy();
                if entropy > 0 && entropy < min_entropy {
                    min_entropy = entropy;
                    best_pos = Some((x, y));
                }
            }
        }
        
        best_pos
    }
    
    /// Propagate constraints from a collapsed cell to its neighbors
    /// 
    /// **Learning Point**: When a cell is collapsed, we must remove incompatible
    /// tile types from neighboring cells' wave functions. This is constraint propagation.
    fn propagate_constraints(&mut self, x: i32, y: i32) {
        let tile = match self.grid[y as usize][x as usize] {
            Some(t) => t,
            None => return,
        };
        
        let edges = get_tile_edges(tile);
        
        // Propagate to north neighbor
        if y > 0 {
            let neighbor_y = y - 1;
            let neighbor_x = x;
            if self.grid[neighbor_y as usize][neighbor_x as usize].is_none() {
                let mut changed = false;
                let possible_tiles = self.wave[neighbor_y as usize][neighbor_x as usize].possible_tiles.clone();
                for &possible_tile in &possible_tiles {
                    let possible_edges = get_tile_edges(possible_tile);
                    if !edges_compatible(possible_edges.south, edges.north) {
                        self.wave[neighbor_y as usize][neighbor_x as usize].remove_tile(possible_tile);
                        changed = true;
                    }
                }
                if changed {
                    self.propagate_constraints(neighbor_x, neighbor_y);
                }
            }
        }
        
        // Propagate to south neighbor
        if y < self.height - 1 {
            let neighbor_y = y + 1;
            let neighbor_x = x;
            if self.grid[neighbor_y as usize][neighbor_x as usize].is_none() {
                let mut changed = false;
                let possible_tiles = self.wave[neighbor_y as usize][neighbor_x as usize].possible_tiles.clone();
                for &possible_tile in &possible_tiles {
                    let possible_edges = get_tile_edges(possible_tile);
                    if !edges_compatible(possible_edges.north, edges.south) {
                        self.wave[neighbor_y as usize][neighbor_x as usize].remove_tile(possible_tile);
                        changed = true;
                    }
                }
                if changed {
                    self.propagate_constraints(neighbor_x, neighbor_y);
                }
            }
        }
        
        // Propagate to east neighbor
        if x < self.width - 1 {
            let neighbor_y = y;
            let neighbor_x = x + 1;
            if self.grid[neighbor_y as usize][neighbor_x as usize].is_none() {
                let mut changed = false;
                let possible_tiles = self.wave[neighbor_y as usize][neighbor_x as usize].possible_tiles.clone();
                for &possible_tile in &possible_tiles {
                    let possible_edges = get_tile_edges(possible_tile);
                    if !edges_compatible(possible_edges.west, edges.east) {
                        self.wave[neighbor_y as usize][neighbor_x as usize].remove_tile(possible_tile);
                        changed = true;
                    }
                }
                if changed {
                    self.propagate_constraints(neighbor_x, neighbor_y);
                }
            }
        }
        
        // Propagate to west neighbor
        if x > 0 {
            let neighbor_y = y;
            let neighbor_x = x - 1;
            if self.grid[neighbor_y as usize][neighbor_x as usize].is_none() {
                let mut changed = false;
                let possible_tiles = self.wave[neighbor_y as usize][neighbor_x as usize].possible_tiles.clone();
                for &possible_tile in &possible_tiles {
                    let possible_edges = get_tile_edges(possible_tile);
                    if !edges_compatible(possible_edges.east, edges.west) {
                        self.wave[neighbor_y as usize][neighbor_x as usize].remove_tile(possible_tile);
                        changed = true;
                    }
                }
                if changed {
                    self.propagate_constraints(neighbor_x, neighbor_y);
                }
            }
        }
    }
}

static WFC_STATE: LazyLock<Mutex<WfcState>> = LazyLock::new(|| Mutex::new(WfcState::new()));

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Generate a new WFC layout using Voronoi noise for grass + proper WFC
/// 
/// **Learning Point**: This implements a two-phase WFC algorithm:
/// Phase 1: Generate Voronoi diagram to create natural-looking grass regions
/// Phase 2: Use proper WFC with entropy tracking to fill remaining cells
/// 
/// The algorithm:
/// 1. Generate Voronoi grass regions (pre-constraints)
/// 2. Initialize wave function for all cells
/// 3. Pre-collapse grass cells based on Voronoi
/// 4. Iteratively find cells with lowest entropy
/// 5. Collapse cells and propagate constraints
/// 6. Repeat until complete
#[wasm_bindgen]
pub fn generate_layout() {
    let mut state = WFC_STATE.lock().unwrap();
    state.clear();
    
    // Phase 1: Generate Voronoi grass regions
    // Use 10 seed points for a 50x50 grid (adjustable for different grass density)
    let grass_map = generate_voronoi_grass(state.width, state.height, 10);
    
    // Phase 2: Initialize wave function and apply pre-constraints
    for y in 0..state.height {
        for x in 0..state.width {
            // Check if there's a pre-constraint for this cell
            if let Some(pre_tile) = state.pre_constraints[y as usize][x as usize] {
                // Pre-collapse cell with pre-constraint
                state.grid[y as usize][x as usize] = Some(pre_tile);
                state.wave[y as usize][x as usize].possible_tiles = vec![pre_tile];
            } else if grass_map[y as usize][x as usize] {
                // Pre-collapse grass cells from Voronoi
                state.grid[y as usize][x as usize] = Some(TileType::Grass);
                state.wave[y as usize][x as usize].possible_tiles = vec![TileType::Grass];
            } else {
                // For non-grass cells, filter out grass from possibilities
                state.wave[y as usize][x as usize].remove_tile(TileType::Grass);
            }
        }
    }
    
    // Propagate constraints from pre-collapsed grass cells
    for y in 0..state.height {
        for x in 0..state.width {
            if state.grid[y as usize][x as usize].is_some() {
                state.propagate_constraints(x, y);
            }
        }
    }
    
    // Phase 3: WFC collapse loop
    // Continue until all cells are collapsed
    loop {
        // Find cell with lowest entropy
        let Some((x, y)) = state.find_lowest_entropy() else {
            // No more cells with valid entropy found
            // Fill any remaining uncollapsed cells to prevent gaps
            for y in 0..state.height {
                for x in 0..state.width {
                    if state.grid[y as usize][x as usize].is_none() {
                        // Cell is still uncollapsed - fill with floor as fallback
                        state.grid[y as usize][x as usize] = Some(TileType::Floor);
                        state.wave[y as usize][x as usize].possible_tiles = vec![TileType::Floor];
                        state.propagate_constraints(x, y);
                    }
                }
            }
            break;
        };
        
        // Collapse the cell
        if let Some(tile) = state.wave[y as usize][x as usize].collapse() {
            state.grid[y as usize][x as usize] = Some(tile);
            // Propagate constraints to neighbors
            state.propagate_constraints(x, y);
        } else {
            // Contradiction - no valid tiles (shouldn't happen with proper WFC)
            // Fallback to floor
            state.grid[y as usize][x as usize] = Some(TileType::Floor);
            state.wave[y as usize][x as usize].possible_tiles = vec![TileType::Floor];
            state.propagate_constraints(x, y);
        }
    }
}

/// Get tile type at a specific grid position
/// 
/// **Learning Point**: This function is called from TypeScript to get the tile
/// at a specific position for rendering. Returns -1 if position is invalid or empty.
/// 
/// @param x - Grid X coordinate (0-49)
/// @param y - Grid Y coordinate (0-49)
/// @returns Tile type as i32, or -1 if invalid/empty
#[wasm_bindgen]
pub fn get_tile_at(x: i32, y: i32) -> i32 {
    let state = WFC_STATE.lock().unwrap();
    if let Some(tile) = state.get_tile(x, y) {
        tile as i32
    } else {
        -1
    }
}

/// Clear the current layout
/// 
/// **Learning Point**: This resets the grid to all empty cells. Called when
/// the user clicks "Recompute Wave Collapse" to start fresh.
#[wasm_bindgen]
pub fn clear_layout() {
    let mut state = WFC_STATE.lock().unwrap();
    state.clear();
}

/// Set a pre-constraint at a specific position
/// 
/// **Learning Point**: Pre-constraints allow external systems (like GPT-2 text-to-layout)
/// to set specific tiles before WFC runs. This enables guided generation based on
/// high-level layout descriptions.
/// 
/// @param x - Grid X coordinate (0-49)
/// @param y - Grid Y coordinate (0-49)
/// @param tile_type - Tile type as i32 (0-10, matching TileType enum)
/// @returns true if constraint was set successfully, false if coordinates are invalid
#[wasm_bindgen]
pub fn set_pre_constraint(x: i32, y: i32, tile_type: i32) -> bool {
    let mut state = WFC_STATE.lock().unwrap();
    
    // Convert i32 to TileType
    let tile = match tile_type {
        0 => TileType::Grass,
        1 => TileType::Floor,
        2 => TileType::WallNorth,
        3 => TileType::WallSouth,
        4 => TileType::WallEast,
        5 => TileType::WallWest,
        6 => TileType::CornerNE,
        7 => TileType::CornerNW,
        8 => TileType::CornerSE,
        9 => TileType::CornerSW,
        10 => TileType::Door,
        _ => return false, // Invalid tile type
    };
    
    state.set_pre_constraint(x, y, tile)
}

/// Clear all pre-constraints
/// 
/// **Learning Point**: This clears all pre-constraints, allowing WFC to generate
/// completely random layouts again. Useful for resetting after text-guided generation.
#[wasm_bindgen]
pub fn clear_pre_constraints() {
    let mut state = WFC_STATE.lock().unwrap();
    state.clear_pre_constraints();
}

/// JavaScript random number generator
/// 
/// **Learning Point**: WASM can't generate random numbers directly, so we
/// call back to JavaScript's Math.random(). This is set up in the TypeScript code.
/// The function is attached to globalThis in the TypeScript route handler.
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = "js_random")]
    fn js_random() -> f64;
}
