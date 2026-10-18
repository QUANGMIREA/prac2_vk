from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

BASE = Path(__file__).resolve().parent


def read_people(path: Path, *, required: bool = True) -> dict[str, str]:
    if not path.exists():
        if not required:
            print(f'Note: {path.name} not found; outside users will display by VK ID.')
            return {}
        raise FileNotFoundError(path)
    people: dict[str, str] = {}
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {'vk_id', 'name'} <= set(reader.fieldnames):
            raise ValueError(f'{path.name}: expected columns vk_id,name')
        for line_no, row in enumerate(reader, start=2):
            uid = (row.get('vk_id') or '').strip()
            name = (row.get('name') or '').strip()
            if not uid or not name:
                raise ValueError(f'{path.name}:{line_no}: empty vk_id or name')
            if uid in people:
                raise ValueError(f'{path.name}:{line_no}: duplicate vk_id {uid}')
            people[uid] = name
    if required and not people:
        raise ValueError(f'{path.name}: no class members found')
    return people


def load_graph(path: Path, members: dict[str, str]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(members)  # retain isolated class members
    duplicates = 0
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {'source', 'target'} <= set(reader.fieldnames):
            raise ValueError(f'{path.name}: expected columns source,target')
        for line_no, row in enumerate(reader, start=2):
            source = (row.get('source') or '').strip()
            target = (row.get('target') or '').strip()
            if not source or not target:
                raise ValueError(f'{path.name}:{line_no}: empty source or target')
            if source == target:
                print(f'Warning: ignored self-link on line {line_no}: {source}')
                continue
            if graph.has_edge(source, target):
                duplicates += 1
            graph.add_edge(source, target)
    if duplicates:
        print(f'Note: ignored {duplicates} duplicate/reversed edge(s).')
    return graph


def calculate(graph: nx.Graph, members: dict[str, str]) -> list[dict]:
    between = nx.betweenness_centrality(graph, normalized=True)
    close = nx.closeness_centrality(graph, wf_improved=True)
    # Scores are computed independently inside each connected component.
    # Values from DIFFERENT components should not be ranked against each other.
    eigen: dict[str, float] = {}
    for component in nx.connected_components(graph):
        sub = graph.subgraph(component)
        if len(sub) == 1:
            eigen[next(iter(component))] = 0.0
        else:
            eigen.update(nx.eigenvector_centrality(sub, max_iter=3000, tol=1e-10))
    return [{
        'vk_id': uid,
        'name': name,
        'degree': graph.degree(uid),
        'betweenness': between[uid],
        'closeness': close[uid],
        'eigenvector': eigen[uid],
    } for uid, name in members.items()]


def save_results(rows: list[dict], path: Path) -> None:
    columns = ['vk_id', 'name', 'degree', 'betweenness', 'closeness', 'eigenvector']
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f'{row[k]:.6f}' if isinstance(row[k], float) else row[k]
                             for k in columns})


def draw_graph(graph: nx.Graph, members: dict[str, str], others: dict[str, str], path: Path) -> None:
    plt.rcParams['font.family'] = 'DejaVu Sans'  # supports Cyrillic
    count = graph.number_of_nodes()
    fig, ax = plt.subplots(figsize=(max(13, min(19, count * 0.33)), 10))
    pos = nx.spring_layout(graph, seed=42, iterations=250, k=1.8 / max(count, 1) ** 0.5)
    class_nodes = [n for n in graph if n in members]
    outer_nodes = [n for n in graph if n not in members]
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='#777777', alpha=0.25, width=0.8)
    if outer_nodes:
        nx.draw_networkx_nodes(graph, pos, nodelist=outer_nodes, ax=ax,
                               node_color='#74c8af', node_size=430,
                               edgecolors='#22614f', linewidths=0.7)
    nx.draw_networkx_nodes(graph, pos, nodelist=class_nodes, ax=ax,
                           node_color='#ffac46', node_size=800,
                           edgecolors='#8a4b00', linewidths=1.0)
    labels = {n: members.get(n, others.get(n, n)) for n in graph}
    # Offset labels slightly above nodes to reduce overlapping with node circles.
    label_pos = {n: (x, y + 0.025) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(graph, label_pos, labels=labels, font_size=7.5,
                            font_family='DejaVu Sans', ax=ax)
    ax.set_title('Сеть друзей ВК: оранжевый — группа, зелёный — другие пользователи', fontsize=13)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def draw_centralities(rows: list[dict], path: Path) -> None:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    names = [r['name'] for r in rows]
    metrics = [('betweenness', 'Посредничество'),
               ('closeness', 'Близость'),
               ('eigenvector', 'Собственный вектор')]
    fig, axes = plt.subplots(1, 3, figsize=(19, max(6, len(rows) * 0.42)))
    for ax, (key, title) in zip(axes, metrics):
        ax.barh(names, [r[key] for r in rows], color='#ed9a32')
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel('Значение')
        ax.grid(axis='x', alpha=0.15)
    fig.suptitle('Центральность участников группы (по всей наблюдаемой сети)')
    fig.tight_layout()
    fig.savefig(path, dpi=175, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='VK social graph: centrality of class members')
    parser.add_argument('--members', type=Path, default=BASE / 'members.csv')
    parser.add_argument('--other-users', type=Path, default=BASE / 'other_users.csv')
    parser.add_argument('--edges', type=Path, default=BASE / 'edges.csv')
    parser.add_argument('--out', type=Path, default=BASE / 'output')
    args = parser.parse_args()

    members = read_people(args.members)
    others = read_people(args.other_users, required=False)
    overlap = set(members) & set(others)
    if overlap:
        raise ValueError('Users in both members.csv and other_users.csv: ' + ', '.join(sorted(overlap)))
    graph = load_graph(args.edges, members)
    not_connected = [members[n] for n in members if graph.degree(n) == 0]
    if not_connected:
        print('Warning: members with no observed connections: ' + ', '.join(not_connected))
    unknown = set(graph) - set(members) - set(others)
    if unknown:
        print('Note: users without display names: ' + ', '.join(sorted(unknown)))
    extras = set(others) - set(graph)
    if extras:
        print(f'Note: {len(extras)} other_users.csv account(s) do not appear in edges.csv.')

    args.out.mkdir(parents=True, exist_ok=True)
    rows = calculate(graph, members)
    save_results(rows, args.out / 'centrality_results.csv')
    draw_graph(graph, members, others, args.out / 'network.png')
    draw_centralities(rows, args.out / 'centrality_chart.png')

    print(f'Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges')
    print(f'Class members: {len(members)}; outside nodes: {len(set(graph) - set(members))}')
    print(f'Connected components: {nx.number_connected_components(graph)}')
    print('\nMember | Degree | Betweenness | Closeness | Eigenvector')
    for row in rows:
        print(f"{row['name']} | {row['degree']} | {row['betweenness']:.4f} | "
              f"{row['closeness']:.4f} | {row['eigenvector']:.4f}")
    print(f'\nSaved files in: {args.out.resolve()}')
    if nx.number_connected_components(graph) > 1:
        print('Warning: eigenvector centrality was calculated per component; do not compare across components.')


if __name__ == '__main__':
    main()
