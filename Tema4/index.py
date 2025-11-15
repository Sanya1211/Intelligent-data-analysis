import random
import collections
import math
import statistics

import networkx as nx
import matplotlib.pyplot as plt


# ============================
# 1. Завантаження ПІДГРАФА
# ============================

def load_friendster_subgraph(edge_path, max_edges=1_000_000):
    """
    Завантажує підграф Friendster з перших max_edges ребер файлу.
    Формат рядка у файлі: "u v"
    Коментарі (рядки з '#') пропускаються.
    Працює зі ЗВИЧАЙНИМ .txt файлом (без gzip).
    """
    G = nx.Graph()
    with open(edge_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u_str, v_str = parts[:2]
            u, v = int(u_str), int(v_str)
            G.add_edge(u, v)
            if max_edges is not None and (i + 1) >= max_edges:
                break
    return G


# ============================
# 2. Завантаження СПІЛЬНОТ
# ============================

def load_communities(cmty_path, max_communities=None, min_size=3):
    """
    Завантажує спільноти з .cmty.txt файлу.
    Кожен рядок: список node_id через пробіл.
    Повертає список списків вершин.
    Працює зі ЗВИЧАЙНИМ .txt файлом (без gzip).
    """
    communities = []
    with open(cmty_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            nodes = list(map(int, line.split()))
            if len(nodes) >= min_size:
                communities.append(nodes)
                if max_communities is not None and len(communities) >= max_communities:
                    break
    return communities


# ============================
# 3. БАЗОВІ ХАРАКТЕРИСТИКИ ГРАФА
# ============================

def basic_graph_stats(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    avg_degree = 2 * m / n if n > 0 else 0

    # степені вершин
    degrees = [deg for _, deg in G.degree()]
    max_degree = max(degrees) if degrees else 0
    min_degree = min(degrees) if degrees else 0

    print("=== Basic Graph Stats ===")
    print(f"Nodes: {n}")
    print(f"Edges: {m}")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Min degree: {min_degree}")
    print(f"Max degree: {max_degree}")

    # Розподіл степенів (кількість вершин з k зв'язками)
    degree_counts = collections.Counter(degrees)
    print("\nDegree distribution (sample):")
    for k in sorted(degree_counts)[:10]:
        print(f"degree {k}: {degree_counts[k]} nodes")

    return {
        "n": n,
        "m": m,
        "avg_degree": avg_degree,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "degree_counts": degree_counts,
    }


# ============================
# 4. BFS / DFS аналіз
# ============================

def bfs_dfs_demo(G, source=None, depth_limit=3):
    if source is None:
        source = next(iter(G.nodes))

    print(f"\n=== BFS/DFS from source {source} (depth_limit={depth_limit}) ===")

    # BFS
    bfs_edges = list(nx.bfs_edges(G, source=source, depth_limit=depth_limit))
    bfs_tree = nx.bfs_tree(G, source=source, depth_limit=depth_limit)
    print(f"BFS edges count: {len(bfs_edges)}")
    print(f"BFS tree nodes: {bfs_tree.number_of_nodes()}")

    # Можемо подивитися шари BFS
    print("\nBFS layers:")
    for layer_id, layer in enumerate(nx.bfs_layers(G, [source])):
        print(f"Layer {layer_id}: {len(layer)} nodes")
        if layer_id >= depth_limit:
            break

    # DFS
    dfs_edges = list(nx.dfs_edges(G, source=source, depth_limit=depth_limit))
    dfs_tree = nx.dfs_tree(G, source=source, depth_limit=depth_limit)
    print(f"\nDFS edges count: {len(dfs_edges)}")
    print(f"DFS tree nodes: {dfs_tree.number_of_nodes()}")

    # Приклади dfs_preorder_nodes / dfs_postorder_nodes
    preorder = list(nx.dfs_preorder_nodes(G, source=source, depth_limit=depth_limit))
    postorder = list(nx.dfs_postorder_nodes(G, source=source, depth_limit=depth_limit))
    print(f"\nDFS preorder (sample): {preorder[:10]}")
    print(f"DFS postorder (sample): {postorder[:10]}")

    return {
        "bfs_tree": bfs_tree,
        "dfs_tree": dfs_tree,
        "bfs_edges": bfs_edges,
        "dfs_edges": dfs_edges,
    }


# ============================
# 5. КОМПОНЕНТИ ЗВ’ЯЗНОСТІ,
#    ДІАМЕТР, ШЛЯХИ
# ============================

def connectivity_and_paths(G, sample_size=10):
    print("\n=== Connectivity & Paths (approx) ===")

    # Найбільша компонента зв’язності
    if nx.is_empty(G):
        print("Graph is empty.")
        return

    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)
    giant = G.subgraph(components[0]).copy()
    print(f"Largest component size: {giant.number_of_nodes()} nodes, {giant.number_of_edges()} edges")

    # Наближення діаметра через BFS з кількох випадкових вершин
    nodes_list = list(giant.nodes())
    sample_nodes = random.sample(nodes_list, min(sample_size, len(nodes_list)))

    ecc_estimates = []
    for s in sample_nodes:
        lengths = nx.single_source_shortest_path_length(giant, s)
        ecc = max(lengths.values())
        ecc_estimates.append(ecc)

    approx_diameter = max(ecc_estimates) if ecc_estimates else 0
    print(f"Approximate diameter (on sample of {len(sample_nodes)} nodes): {approx_diameter}")

    return {
        "largest_component": giant,
        "approx_diameter": approx_diameter,
        "ecc_estimates": ecc_estimates,
    }


# ============================
# 6. КЛАСТЕРИЗАЦІЯ, ЦЕНТРАЛЬНІСТЬ
# ============================

def clustering_and_centrality(G, sample_size=5000):
    print("\n=== Clustering & Centrality (approx) ===")

    # якщо граф великий – беремо підвибірку вершин
    nodes_list = list(G.nodes())
    if len(nodes_list) > sample_size:
        sample_nodes = set(random.sample(nodes_list, sample_size))
        H = G.subgraph(sample_nodes).copy()
        print(f"Using induced subgraph of {len(H)} nodes for clustering/centrality.")
    else:
        H = G

    # середній коефіцієнт кластеризації
    avg_clustering = nx.average_clustering(H)
    print(f"Average clustering coefficient (sample): {avg_clustering:.4f}")

    # degree centrality
    deg_cent = nx.degree_centrality(H)
    top_deg = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop-10 nodes by degree centrality:")
    for node, val in top_deg:
        print(f"node {node}: {val:.5f}")

    # приближена міжпосередницька центральність (betweenness)
    bet_cent = nx.betweenness_centrality(H, k=min(100, H.number_of_nodes()), normalized=True, seed=42)
    top_bet = sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop-10 nodes by betweenness centrality:")
    for node, val in top_bet:
        print(f"node {node}: {val:.5f}")

    return {
        "avg_clustering": avg_clustering,
        "degree_centrality": deg_cent,
        "betweenness_centrality": bet_cent,
    }


# ============================
# 7. АНАЛІЗ СПІЛЬНОТ
# ============================

def communities_stats(communities, G=None):
    print("\n=== Community Stats ===")
    sizes = [len(c) for c in communities]
    print(f"Total communities loaded: {len(communities)}")
    print(f"Min community size: {min(sizes)}")
    print(f"Max community size: {max(sizes)}")
    print(f"Average community size: {statistics.mean(sizes):.2f}")

    # Приклад: щільність кількох спільнот (якщо переданий граф G)
    if G is not None:
        print("\nExample of community density for first 5 communities:")
        for i, comm in enumerate(communities[:5]):
            sub = G.subgraph(comm)
            density = nx.density(sub)
            print(f"Community {i}, size {len(comm)}, density {density:.4f}")


# ============================
# 8. ЛЕГКА ВІЗУАЛІЗАЦІЯ
# ============================

def visualize_small_subgraph(G, num_nodes=100):
    """
    Візуалізуємо дуже маленький підграф для прикладу (бо великий малювати безглуздо).
    """
    if G.number_of_nodes() <= num_nodes:
        H = G
    else:
        sampled = set(random.sample(list(G.nodes()), num_nodes))
        H = G.subgraph(sampled).copy()

    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(H)
    nx.draw(H, pos, node_size=30, with_labels=False)
    plt.title(f"Visualization of subgraph with {H.number_of_nodes()} nodes")
    plt.show()


# ============================
# 9. MAIN: запуск аналізу
# ============================

if __name__ == "__main__":
    # Імена файлів ТАКІ САМІ, як у тебе в папці Tema4
    EDGE_FILE = "com-friendster.ungraph.txt"
    CMTY_FILE = "com-friendster.top5000.cmty.txt"

    # 1) Завантажуємо підграф
    print("Loading Friendster subgraph...")
    G = load_friendster_subgraph(EDGE_FILE, max_edges=1_000_000)

    # 2) Базові характеристики графа
    stats = basic_graph_stats(G)

    # 3) BFS/DFS демонстрація
    bfs_dfs_results = bfs_dfs_demo(G, depth_limit=3)

    # 4) Компоненти зв’язності та наближений діаметр
    conn_results = connectivity_and_paths(G, sample_size=5)

    # 5) Кластеризація і центральності
    cc_results = clustering_and_centrality(G, sample_size=5000)

    # 6) Аналіз спільнот
    print("\nLoading communities...")
    communities = load_communities(CMTY_FILE, max_communities=5000, min_size=3)
    communities_stats(communities, G=G)

    # 7) Візуалізація маленького фрагмента графа
    visualize_small_subgraph(G, num_nodes=150)
