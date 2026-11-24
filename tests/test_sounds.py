from airpoint.sounds import FILES, Sounds


def test_each_action_has_its_own_system_sound():
    played = []
    s = Sounds(player=played.append)
    for kind in ("click", "right_click", "drag_start", "drag_end", "lock", "unlock"):
        s.play(kind)
    assert [p.rsplit("/", 1)[1] for p in played] == [FILES[k] + ".aiff" for k in
                                                     ("click", "right_click", "drag_start", "drag_end", "lock", "unlock")]


def test_scroll_and_unknown_kinds_are_silent_and_off_means_off():
    played = []
    s = Sounds(player=played.append)
    s.play("scroll")
    s.play("hotkey")
    s.enabled = False
    s.play("click")
    assert played == []
