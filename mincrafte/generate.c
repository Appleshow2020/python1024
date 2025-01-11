#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

// Constants for initial setup
#define SEA_SIZE (1LL << 31) // Logical size 2^31

// Terrain types
typedef enum {
    SEA = 0,
    LAND,
    MUSHROOM_ISLAND,
    ICE_BIOME,
    COLD_BIOME,
    WARM_BIOME,
    SPECIAL_BIOME,
    TUNDRA,
    SNOWY_TAIGA,
    FOREST,
    MOUNTAIN,
    PLAINS,
    TAIGA,
    DESERT,
    SAVANNA,
    DARK_FOREST,
    SWAMP,
    BIRCH_FOREST,
    JUNGLE,
    BAMBOO_FOREST,
    SUNFLOWER_PLAINS,
    MESA,
    GIANT_SPRUCE_FOREST,
    DEEP_OCEAN,
    WARM_OCEAN,
    LUKEWARM_OCEAN,
    COLD_OCEAN,
    FROZEN_OCEAN,
    BEACH,
    RIVER
} TerrainType;

// Memory allocation for map
TerrainType** allocate_map(int64_t width, int64_t height) {
    TerrainType** map = (TerrainType**)malloc(height * sizeof(TerrainType*));
    for (int64_t i = 0; i < height; i++) {
        map[i] = (TerrainType*)calloc(width, sizeof(TerrainType));
    }
    return map;
}

// Free map memory
void free_map(TerrainType** map, int64_t height) {
    for (int64_t i = 0; i < height; i++) {
        free(map[i]);
    }
    free(map);
}

// Initialize the sea with 10% chance of land generation
void initialize_sea(TerrainType** map, int64_t width, int64_t height) {
    for (int64_t i = 0; i < height; i++) {
        for (int64_t j = 0; j < width; j++) {
            map[i][j] = (rand() % 100 < 10) ? LAND : SEA;
        }
    }
}

// Expand the map by doubling its size
TerrainType** expand_map(TerrainType** map, int64_t* width, int64_t* height) {
    int64_t new_width = (*width) * 2;
    int64_t new_height = (*height) * 2;

    TerrainType** new_map = allocate_map(new_width, new_height);

    for (int64_t i = 0; i < *height; i++) {
        for (int64_t j = 0; j < *width; j++) {
            TerrainType current = map[i][j];

            new_map[i * 2][j * 2] = current;
            new_map[i * 2 + 1][j * 2] = (rand() % 100 < 60) ? current : SEA;
            new_map[i * 2][j * 2 + 1] = (rand() % 100 < 60) ? current : SEA;
            new_map[i * 2 + 1][j * 2 + 1] = (rand() % 100 < 30) ? current : SEA;
        }
    }

    free_map(map, *height);

    *width = new_width;
    *height = new_height;
    return new_map;
}

// Add small islands around existing land
void add_small_islands(TerrainType** map, int64_t width, int64_t height) {
    for (int64_t i = 0; i < height; i++) {
        for (int64_t j = 0; j < width; j++) {
            if (map[i][j] == LAND) {
                if (i > 0 && j > 0 && rand() % 100 < 10) map[i - 1][j - 1] = LAND;
                if (i > 0 && rand() % 100 < 10) map[i - 1][j] = LAND;
                if (i > 0 && j < width - 1 && rand() % 100 < 10) map[i - 1][j + 1] = LAND;
                if (j > 0 && rand() % 100 < 10) map[i][j - 1] = LAND;
                if (j < width - 1 && rand() % 100 < 10) map[i][j + 1] = LAND;
                if (i < height - 1 && j > 0 && rand() % 100 < 10) map[i + 1][j - 1] = LAND;
                if (i < height - 1 && rand() % 100 < 10) map[i + 1][j] = LAND;
                if (i < height - 1 && j < width - 1 && rand() % 100 < 10) map[i + 1][j + 1] = LAND;
            }
        }
    }
}

// Add mushroom islands randomly in surrounded sea
void add_mushroom_islands(TerrainType** map, int64_t width, int64_t height) {
    for (int64_t i = 1; i < height - 1; i++) {
        for (int64_t j = 1; j < width - 1; j++) {
            if (map[i][j] == SEA) {
                bool surrounded_by_sea = true;
                for (int64_t di = -1; di <= 1; di++) {
                    for (int64_t dj = -1; dj <= 1; dj++) {
                        if (map[i + di][j + dj] != SEA) {
                            surrounded_by_sea = false;
                            break;
                        }
                    }
                    if (!surrounded_by_sea) break;
                }
                if (surrounded_by_sea && rand() % 100 < 1) {
                    map[i][j] = MUSHROOM_ISLAND;
                }
            }
        }
    }
}

// Set up the climate zones
void setup_climate(TerrainType** map, int64_t width, int64_t height) {
    for (int64_t i = 0; i < height; i++) {
        for (int64_t j = 0; j < width; j++) {
            int random_value = rand() % 6;
            if (random_value == 0) {
                map[i][j] = ICE_BIOME;
            } else if (random_value == 1) {
                map[i][j] = COLD_BIOME;
            } else {
                map[i][j] = WARM_BIOME;
            }
        }
    }
}

// Adjust biomes based on neighboring climates
void adjust_biomes(TerrainType** map, int64_t width, int64_t height) {
    for (int64_t i = 1; i < height - 1; i++) {
        for (int64_t j = 1; j < width - 1; j++) {
            if (map[i][j] == WARM_BIOME) {
                for (int64_t di = -1; di <= 1; di++) {
                    for (int64_t dj = -1; dj <= 1; dj++) {
                        if (map[i + di][j + dj] == COLD_BIOME) {
                            map[i][j] = SPECIAL_BIOME; // Intermediate biome
                            break;
                        }
                    }
                }
            }
            if (map[i][j] == ICE_BIOME) {
                for (int64_t di = -1; di <= 1; di++) {
                    for (int64_t dj = -1; dj <= 1; dj++) {
                        if (map[i + di][j + dj] == WARM_BIOME) {
                            map[i][j] = COLD_BIOME;
                            break;
                        }
                    }
                }
            }
        }
    }
}

// Split land into smaller islands
void split_land_into_islands(TerrainType** map, int64_t width, int64_t height) {
    for (int64_t i = 0; i < height; i++) {
        for (int64_t j = 0; j < width; j++) {
            if (map[i][j] == LAND && rand() % 100 < 20) {
                map[i][j] = SEA;
            }
        }
    }
}

// Set deep ocean for isolated water bodies
void set_deep_ocean(TerrainType** map, int64_t width, int64_t height) {
    for (int64_t i = 1; i < height - 1; i++) {
        for (int64_t j = 1; j < width - 1; j++) {
            if (map[i][j] == SEA) {
                bool surrounded_by_sea = true;
                for (int64_t di = -1; di <= 1; di++) {
                    for (int64_t dj = -1; dj <= 1; dj++) {
                        if (map[i + di][j + dj] != SEA) {
                            surrounded_by_sea = false;
                            break;
                        }
                    }
                    if (!surrounded_by_sea) break;
                }
                if (surrounded_by_sea) {
                    map[i][j] = DEEP_OCEAN;
                }
            }
        }
    }
}

void make_beach(TerrainType** map, int64_t width, int64_t height) {
    for (int64_t i=1; i<height-1; i++) {
        for (int64_t j=1; j<width-1; j++) {
            if (map[i][j] == SEA) {
                if (map[i][j+1] == LAND) {
                    map[i][j] = BEACH;
                }
            } else if (map[i][j] == LAND) {
                if (map[i][j+1] == SEA) {
                    map[i][j] = BEACH;
                }
            }
        }
    }
}

void split_into_two(TerrainType** map, int64_t width, int64_t height) {
    for (int64_t i=1; i<height-1; i++) {
        for (int64_t j=1; j<height-1; j++) {
            if (map[i][j] != SEA) {
                if (rand()%2 == 0) {
                    map[i][j] == true;
                } else {
                    map[i][j] == false;
                }
            }
        }
    }
}

void make_river(TerrainType** map, int64_t width, int64_t height) {
    for (int64_t i=1; i<height-1; i++) {
        for (int64_t j=1; j<width-1; j++) {
            if (map[i][j]) {
                if !(map[i][j+1]) {
                    map[i][j] = RIVER;
                }
            } else if !(map[i][j]) {
                if (map[i][j+1]) {
                    map[i][j] = RIVER;
                }
            }
            
        }
    }
}



// Main function
int main() {
    srand(time(NULL)); // Initialize random seed

    // Set initial size based on block size
    int64_t block_size = 1024 * 1024;
    int64_t width = SEA_SIZE / block_size;
    int64_t height = SEA_SIZE / block_size;

    // Allocate and initialize map
    TerrainType** map = allocate_map(width, height);
    initialize_sea(map, width, height);

    // Step-by-step algorithm execution
    map = expand_map(map, &width, &height);
    add_small_islands(map, width, height);
    add_mushroom_islands(map, width, height);
    add_small_islands(map,width,height);
    setup_climate(map, width, height);
    adjust_biomes(map, width, height);
    map = expand_map(map, &width, &height);
    split_land_into_islands(map, width, height);
    set_deep_ocean(map, width, height);
    map2 = map;
    width2= width;
    height2 = height;
    // TODO : step13
    map = expand_map(map, &width, &height);
    map = expand_map(map, &width, &height);
    map = expand_map(map, &width, &height);
    make_beach(map, width, height);
    map = expand_map(map, &width, &height);
    add_small_islands(map, width, height);
    map = expand_map(map, &width, &height);
    map = expand_map(map, &width, &height);
    
    split_into_two(map2, width2, height2)
    map = expand_map(map, &width, &height);
    map = expand_map(map, &width, &height);
    map = expand_map(map, &width, &height);
    map = expand_map(map, &width, &height);
    map = expand_map(map, &width, &height);
    map = expand_map(map, &width, &height);
    map = expand_map(map, &width, &height);

    // Free allocated memory
    free_map(map, height);
    free_map(map2, height);
    return 0;
}
