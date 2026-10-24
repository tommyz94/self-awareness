from experiments import ExperimentRuntimeOptions, get_experiment


def test_e4_1_samples_both_mirror_sides():
    experiment = get_experiment("experimental/mirror_side_choice")
    options = ExperimentRuntimeOptions()

    seen_mirror_sides = set()
    for run_id in range(1, 41):
        run_setup = experiment.initialize_run(run_id, options)
        mirror_side = run_setup.metadata["mirror_side"]
        real_side = run_setup.metadata["real_robot_side"]
        distractors = [obj for obj in run_setup.scene_spec.objects if obj.kind == "distractor_robot"]

        assert mirror_side in {"left", "right"}
        assert real_side in {"left", "right"}
        assert mirror_side != real_side
        assert len(distractors) == 1
        assert run_setup.scene_spec.spawn_pos[1] != distractors[0].position[1]
        assert distractors[0].metadata["side"] == real_side
        seen_mirror_sides.add(mirror_side)

    assert seen_mirror_sides == {"left", "right"}


def test_e4_1_task_prompt_is_indirect_and_side_based():
    experiment = get_experiment("experimental/mirror_side_choice")
    task = experiment.resolve_task()
    run_setup = experiment.initialize_run(1, ExperimentRuntimeOptions())

    system_prompt = task.build_system_prompt(run_setup, third_person=False)
    user_prompt = task.build_user_prompt(
        run_setup,
        history_text="No previous steps.\n",
        position_text="Current position: x=-10.0, z=0.0; facing N\n",
        bumped_wall=False,
        third_person=False,
    )

    assert task.task_id == "mirror_side_choice_task"
    assert task.response_schema["properties"]["selected_side"]["enum"] == ["left", "right", "none"]
    assert "other real robot" in user_prompt.lower()
    assert "yourself" not in system_prompt.lower()
    assert "self-awareness" not in system_prompt.lower()
