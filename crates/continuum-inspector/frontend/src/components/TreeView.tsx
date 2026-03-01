import { useState, useCallback } from 'preact/hooks';
import type { TreeNode, TreeNodeKind } from '../types/ipc';

/** Badge letter and CSS class for each node kind. */
const KIND_BADGE: Record<TreeNodeKind, { letter: string; cls: string }> = {
  world:     { letter: 'W', cls: 'badge-world' },
  entity:    { letter: 'E', cls: 'badge-entity' },
  signal:    { letter: 'S', cls: 'badge-signal' },
  field:     { letter: 'F', cls: 'badge-field' },
  operator:  { letter: 'O', cls: 'badge-operator' },
  fracture:  { letter: 'X', cls: 'badge-fracture' },
  chronicle: { letter: 'C', cls: 'badge-chronicle' },
  impulse:   { letter: 'I', cls: 'badge-impulse' },
  namespace: { letter: 'N', cls: 'badge-namespace' },
};

interface TreeNodeRowProps {
  node: TreeNode;
  depth: number;
  onSelect: (node: TreeNode) => void;
  expanded: Set<string>;
  toggleExpand: (id: string) => void;
}

function TreeNodeRow({ node, depth, onSelect, expanded, toggleExpand }: TreeNodeRowProps) {
  const hasChildren = node.children.length > 0;
  const isExpanded = expanded.has(node.id);
  const badge = KIND_BADGE[node.kind] || KIND_BADGE.namespace;
  const isLeaf = !hasChildren;

  const handleClick = () => {
    if (isLeaf) {
      onSelect(node);
    } else {
      toggleExpand(node.id);
    }
  };

  const handleBadgeClick = (e: MouseEvent) => {
    // Always select on badge click, even for non-leaf nodes
    e.stopPropagation();
    onSelect(node);
  };

  return (
    <>
      <div
        class={`tree-row ${isLeaf ? 'tree-leaf' : 'tree-branch'}`}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={handleClick}
      >
        {hasChildren && (
          <span class={`tree-chevron ${isExpanded ? 'open' : ''}`}>&#9656;</span>
        )}
        {!hasChildren && <span class="tree-chevron-spacer" />}
        <span class={`tree-badge ${badge.cls}`} onClick={handleBadgeClick}>
          {badge.letter}
        </span>
        <span class="tree-label">{node.label}</span>
        {node.value_type && (
          <span class="tree-type">{node.value_type}</span>
        )}
      </div>
      {isExpanded && node.children.map(child => (
        <TreeNodeRow
          key={child.id}
          node={child}
          depth={depth + 1}
          onSelect={onSelect}
          expanded={expanded}
          toggleExpand={toggleExpand}
        />
      ))}
    </>
  );
}

interface TreeViewProps {
  tree: TreeNode | null;
  onSelect: (node: TreeNode) => void;
}

export function TreeView({ tree, onSelect }: TreeViewProps) {
  const [expanded, setExpanded] = useState<Set<string>>(() => {
    // Auto-expand the root and all entities on first render
    const initial = new Set<string>();
    if (tree) {
      initial.add(tree.id);
      for (const child of tree.children) {
        if (child.kind === 'entity' || child.kind === 'namespace') {
          initial.add(child.id);
        }
      }
    }
    return initial;
  });

  const toggleExpand = useCallback((id: string) => {
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  if (!tree) {
    return <div class="empty">Loading world tree...</div>;
  }

  return (
    <div class="tree-view">
      {tree.children.map(child => (
        <TreeNodeRow
          key={child.id}
          node={child}
          depth={0}
          onSelect={onSelect}
          expanded={expanded}
          toggleExpand={toggleExpand}
        />
      ))}
    </div>
  );
}
