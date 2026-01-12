        // Convert signals to unified nodes
        for (id, signal) in &self.signals {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: signal.file.clone(),
                span: signal.span.clone(),
                stratum: Some(signal.stratum.clone()),
                reads: signal.reads.clone(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Signal(
                    super::unified_nodes::SignalProperties {
                        title: signal.title.clone(),
                        symbol: signal.symbol.clone(),
                        value_type: signal.value_type.clone(),
                        uses_dt_raw: signal.uses_dt_raw,
                        resolve: signal.resolve.clone(),
                        resolve_components: signal.resolve_components.clone(),
                        warmup: signal.warmup.clone(),
                        assertions: signal.assertions.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert fields to unified nodes
        for (id, field) in &self.fields {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: field.file.clone(),
                span: field.span.clone(),
                stratum: Some(field.stratum.clone()),
                reads: field.reads.clone(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Field(
                    super::unified_nodes::FieldProperties {
                        title: field.title.clone(),
                        topology: field.topology,
                        value_type: field.value_type.clone(),
                        measure: field.measure.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert operators to unified nodes
        for (id, operator) in &self.operators {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: operator.file.clone(),
                span: operator.span.clone(),
                stratum: Some(operator.stratum.clone()),
                reads: operator.reads.clone(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Operator(
                    super::unified_nodes::OperatorProperties {
                        phase: operator.phase,
                        body: operator.body.clone(),
                        assertions: operator.assertions.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert impulses to unified nodes
        for (id, impulse) in &self.impulses {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: impulse.file.clone(),
                span: impulse.span.clone(),
                stratum: None,
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Impulse(
                    super::unified_nodes::ImpulseProperties {
                        payload_type: impulse.payload_type.clone(),
                        apply: impulse.apply.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert fractures to unified nodes
        for (id, fracture) in &self.fractures {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: fracture.file.clone(),
                span: fracture.span.clone(),
                stratum: Some(fracture.stratum.clone()),
                reads: fracture.reads.clone(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Fracture(
                    super::unified_nodes::FractureProperties {
                        conditions: fracture.conditions.clone(),
                        emits: fracture.emits.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert entities to unified nodes
        for (id, entity) in &self.entities {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: entity.file.clone(),
                span: entity.span.clone(),
                stratum: None,
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Entity(
                    super::unified_nodes::EntityProperties {
                        count_source: entity.count_source.clone(),
                        count_bounds: entity.count_bounds,
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert members to unified nodes
        for (id, member) in &self.members {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: member.file.clone(),
                span: member.span.clone(),
                stratum: Some(member.stratum.clone()),
                reads: member.reads.clone(),
                member_reads: member.member_reads.clone(),
                kind: super::unified_nodes::NodeKind::Member(
                    super::unified_nodes::MemberProperties {
                        entity_id: member.entity_id.clone(),
                        signal_name: member.signal_name.clone(),
                        title: member.title.clone(),
                        symbol: member.symbol.clone(),
                        value_type: member.value_type.clone(),
                        uses_dt_raw: member.uses_dt_raw,
                        initial: member.initial.clone(),
                        resolve: member.resolve.clone(),
                        assertions: member.assertions.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert chronicles to unified nodes
        for (id, chronicle) in &self.chronicles {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: chronicle.file.clone(),
                span: chronicle.span.clone(),
                stratum: None,
                reads: chronicle.reads.clone(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Chronicle(
                    super::unified_nodes::ChronicleProperties {
                        handlers: chronicle.handlers.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert functions to unified nodes
        for (id, function) in &self.functions {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: function.file.clone(),
                span: function.span.clone(),
                stratum: None,
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Function(
                    super::unified_nodes::FunctionProperties {
                        params: function.params.clone(),
                        body: function.body.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert types to unified nodes
        for (id, ty) in &self.types {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: ty.file.clone(),
                span: ty.span.clone(),
                stratum: None,
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Type(
                    super::unified_nodes::TypeProperties {
                        fields: ty.fields.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert strata to unified nodes
        for (id, stratum) in &self.strata {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: stratum.file.clone(),
                span: stratum.span.clone(),
                stratum: None,
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Stratum(
                    super::unified_nodes::StratumProperties {
                        title: stratum.title.clone(),
                        symbol: stratum.symbol.clone(),
                        default_stride: stratum.default_stride,
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert eras to unified nodes
        for (id, era) in &self.eras {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: era.file.clone(),
                span: era.span.clone(),
                stratum: None,
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Era(
                    super::unified_nodes::EraProperties {
                        is_initial: era.is_initial,
                        is_terminal: era.is_terminal,
                        title: era.title.clone(),
                        dt_seconds: era.dt_seconds,
                        strata_states: era.strata_states.clone(),
                        transitions: era.transitions.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        CompiledWorld {
            constants: self.constants,
            config: self.config,
            nodes,
        }
    }
