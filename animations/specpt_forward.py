from manim import *
import numpy as np

# ── Palette (ManimColor) ─────────────────────────────
BG       = ManimColor("#0E1116")
GOLD     = ManimColor("#FFD166")
BLUE     = ManimColor("#4C9BE8")
PURPLE   = ManimColor("#9B5DE5")
WHITE    = ManimColor("#FFFFFF")
GRAY     = ManimColor("#777777")
TEAL     = ManimColor("#118AB2")
EMERALD  = ManimColor("#06D6A0")
RED_LINE = ManimColor("#EF476F")

# ── Parameters ───────────────────────────────────────
WAV_MIN  = 3600
WAV_MAX  = 9800
N_PATCH  = 12
N_POINTS = 600

np.random.seed(42)

# ── Synthetic spectrum ──────────────────────────────
def synth_spectrum():
    lam = np.linspace(WAV_MIN, WAV_MAX, N_POINTS)
    cont = (0.3 * np.sin((lam - WAV_MIN) / (WAV_MAX - WAV_MIN) * 1.4 * np.pi)
            + 0.6 + 0.15 * np.exp(-((lam - 5000) / 1200) ** 2))
    lines = [(3727, 0.7, 25), (4861, 0.35, 20),
             (5007, 1.0, 18), (6563, 1.4, 30)]
    clean = cont.copy()
    for c, a, w in lines:
        clean += a * np.exp(-((lam - c) / w) ** 2)
    clean = (clean - clean.min()) / (clean.max() - clean.min()) * 3.0 + 0.5
    noisy = clean + 0.06 * np.random.randn(N_POINTS)
    noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min()) * 3.0 + 0.5
    return lam, noisy, clean

# ── Scene ────────────────────────────────────────────
class SpecPTForward(Scene):
    def construct(self):
        self.camera.background_color = BG
        lam, noisy, clean = synth_spectrum()
        patch_edges = np.linspace(WAV_MIN, WAV_MAX, N_PATCH + 1)

        # ════════════════════════════════════════════════
        #  Helpers
        # ════════════════════════════════════════════════
        def patch_data(i):
            """Return (wavelength_center, avg_flux) for patch i."""
            lo, hi = patch_edges[i], patch_edges[i + 1]
            mask = (lam >= lo) & (lam <= hi)
            return 0.5 * (lo + hi), np.mean(noisy[mask])

        def build_spectrum_line(l, f, color=GOLD, sw=2):
            pts = [axes.c2p(l[i], f[i]) for i in range(N_POINTS)]
            v = VMobject(stroke_color=color, stroke_width=sw)
            v.set_points_as_corners(pts)
            return v

        def build_bars(values, n, fill_color, width=0.35, max_h=1.2, center=None):
            vals = values if len(values) == n else [np.mean(values)] * n
            bars = VGroup()
            for v in vals:
                h = max(0.08, v * max_h / 2.5)
                r = Rectangle(width=width, height=h,
                              fill_color=fill_color, fill_opacity=0.85,
                              stroke_color=WHITE, stroke_width=0.5)
                bars.add(r)
            bars.arrange(RIGHT, buff=0.12, aligned_edge=DOWN)
            if center is not None:
                bars.move_to(center)
            return bars

        def sinusoid_curve(lam_range, A=0.3, freq=3.5):
            xs = np.linspace(lam_range[0], lam_range[1], 200)
            pts = [axes.c2p(x, 0.5 + A * np.sin(2 * np.pi * freq * (x - lam_range[0]) / (lam_range[1] - lam_range[0])))
                   for x in xs]
            s = VMobject(stroke_color=GOLD, stroke_width=2, stroke_opacity=0.7)
            s.set_points_as_corners(pts)
            return s

        def attention_weight_12(h=1.0):
            w = np.zeros((12, 12))
            for i in range(12):
                for j in range(12):
                    d = abs(i - j)
                    w[i][j] = h * (0.5 * np.exp(-d / 2.5)
                                   + 0.2 * np.exp(-((i - 3) ** 2 + (j - 5) ** 2) / 8)
                                   + 0.3 * np.exp(-((i - 7) ** 2 + (j - 2) ** 2) / 10)
                                   + 0.05 * np.random.randn())
            return np.clip(w, 0, 1)

        def make_att_grid(w, sz=0.28):
            g = VGroup()
            for i in range(12):
                for j in range(12):
                    v = w[i][j]
                    sq = Square(side_length=sz)
                    sq.set_fill(interpolate_color(BLUE, GOLD, v),
                                opacity=0.2 + 0.8 * v)
                    sq.set_stroke(GRAY, width=0.3)
                    sq.move_to([j * (sz + 0.02), -i * (sz + 0.02), 0])
                    g.add(sq)
            return g

        # ════════════════════════════════════════════════
        #  Layout anchors
        # ════════════════════════════════════════════════
        title   = Text("", font_size=30).to_edge(UP).shift(DOWN * 0.2)
        ax_rect = [-6, -2.2, 6, -1.0]   # bottom strip
        main_center = np.array([0.0, 1.2, 0.0])

        # ──── Persistent axes ────────────────────────
        axes = Axes(
            x_range=[WAV_MIN, WAV_MAX, 1500],
            y_range=[0, 4, 1],
            x_length=12, y_length=2.0,
            axis_config={"color": GRAY, "include_numbers": True,
                         "font_size": 14, "stroke_width": 1},
        ).shift(DOWN * 2.5)
        ax_label_x = axes.get_x_axis_label(Text("Wavelength (Å)", font_size=18, color=GRAY),
                                           edge=DOWN, direction=DOWN, buff=0.25)
        ax_label_y = axes.get_y_axis_label(Text("Flux", font_size=18, color=GRAY),
                                           edge=UP, direction=UP, buff=0.1)

        # ════════════════════════════════════════════════
        #  STAGE 1 — INPUT
        # ════════════════════════════════════════════════
        raw_line = build_spectrum_line(lam, noisy, GOLD, 2)
        label_data = [("[OII]", 3727), ("Hβ", 4861),
                      ("[OIII]", 5007), ("Hα", 6563)]
        labels = VGroup()
        for nm, wl in label_data:
            idx = np.argmin(np.abs(lam - wl))
            lbl = Text(nm, font_size=18, color=GOLD).next_to(
                axes.c2p(wl, noisy[idx]), UP, buff=0.15)
            labels.add(lbl)
        label_arrows = VGroup()
        for nm, wl in label_data:
            idx = np.argmin(np.abs(lam - wl))
            arr = Arrow(axes.c2p(wl, noisy[idx]),
                        axes.c2p(wl, noisy[idx] + 0.6),
                        color=GOLD, stroke_width=1.5, buff=0)
            label_arrows.add(arr)

        t1 = Text("Raw DESI 1D Spectrum", font_size=32).to_edge(UP).shift(DOWN * 0.2)

        self.play(Create(axes),
                  FadeIn(ax_label_x), FadeIn(ax_label_y), run_time=1.5)
        self.play(Create(raw_line), run_time=2)
        self.play(Write(t1), run_time=0.8)
        self.play(LaggedStart(*[GrowArrow(a) for a in label_arrows],
                               *[FadeIn(l, shift=UP * 0.2) for l in labels],
                               lag_ratio=0.15), run_time=2)
        self.wait(1.5)

        # ════════════════════════════════════════════════
        #  STAGE 2 — PATCH EMBEDDING
        # ════════════════════════════════════════════════
        t2 = Text("Patch Embedding", font_size=32).to_edge(UP).shift(DOWN * 0.2)

        dividers = VGroup()
        for pe in patch_edges[1:-1]:
            top = axes.c2p(pe, 4)
            bot = axes.c2p(pe, 0)
            d = DashedLine(bot, top, color=TEAL, stroke_width=1.5,
                           dash_length=0.08, dashed_ratio=0.5)
            dividers.add(d)

        patch_rects = VGroup()
        for i in range(N_PATCH):
            cx, avg = patch_data(i)
            x0 = axes.c2p(patch_edges[i], 0)
            x1 = axes.c2p(patch_edges[i + 1], 0)
            w = x1[0] - x0[0]
            h = max(0.1, avg * 0.35)
            r = Rectangle(width=w, height=h,
                          fill_color=BLUE, fill_opacity=0.18,
                          stroke_color=BLUE, stroke_width=0.8)
            r.move_to([axes.c2p(cx, 0)[0], axes.c2p(0, avg * 0.5)[1], 0])
            patch_rects.add(r)

        embed_bars = build_bars(
            [patch_data(i)[1] for i in range(N_PATCH)],
            N_PATCH, BLUE, width=0.3, max_h=1.0,
            center=main_center + [-1.5, 0, 0])

        self.play(Transform(t1, t2),
                  LaggedStart(*[Create(d) for d in dividers], lag_ratio=0.05),
                  run_time=1.5)
        self.play(LaggedStart(*[Create(r) for r in patch_rects],
                              lag_ratio=0.06), run_time=1.5)

        # morph patch_rects → embed_bars (same VGroup count)
        # Cross-fade + reposition
        for r, b in zip(patch_rects, embed_bars):
            b.move_to(r.get_center())

        self.play(
            LaggedStart(*[Transform(r, b) for r, b in zip(patch_rects, embed_bars)],
                        lag_ratio=0.04),
            FadeOut(dividers, scale=0.5),
            FadeOut(raw_line, shift=DOWN),
            FadeOut(label_arrows), FadeOut(labels),
            run_time=2.5
        )
        self.wait(1)

        # ──── Scale axes down if first time — already there
        # ════════════════════════════════════════════════
        #  STAGE 3 — POSITIONAL ENCODING
        # ════════════════════════════════════════════════
        t3 = Text("Positional Encoding", font_size=32).to_edge(UP).shift(DOWN * 0.2)

        sine = sinusoid_curve([WAV_MIN, WAV_MAX], A=0.25, freq=3.5)
        # Map sine to central area above axes
        sine_bars = VGroup()
        for i, b in enumerate(embed_bars):
            mod = 0.2 * np.sin(2 * np.pi * 3.5 * i / N_PATCH)
            nh = max(0.08, b.height * (1.2 + mod))
            nb = Rectangle(width=0.3, height=nh,
                           fill_color=interpolate_color(BLUE, PURPLE, i / N_PATCH),
                           fill_opacity=0.85,
                           stroke_color=WHITE, stroke_width=0.5)
            nb.move_to(b.get_center())
            sine_bars.add(nb)

        self.play(Transform(t1, t3), Create(sine), run_time=1.5)
        self.play(
            LaggedStart(*[Transform(eb, sb)
                          for eb, sb in zip(embed_bars, sine_bars)],
                        lag_ratio=0.05),
            run_time=2
        )
        self.wait(1)

        # ════════════════════════════════════════════════
        #  STAGE 4 — ENCODER (×3 layers)
        # ════════════════════════════════════════════════
        t4 = Text("Encoder: Multi-Head Self-Attention", font_size=30).to_edge(UP).shift(DOWN * 0.2)

        encoder_bars = VGroup()
        for sb in sine_bars:
            ec = interpolate_color(PURPLE, BLUE, 0.3)
            nb = Rectangle(width=0.28, height=sb.height,
                           fill_color=ec, fill_opacity=0.85,
                           stroke_color=WHITE, stroke_width=0.5)
            nb.move_to(sb.get_center())
            encoder_bars.add(nb)

        self.play(Transform(t1, t4),
                  Transform(sine_bars, encoder_bars),
                  FadeOut(sine, scale=0.5),
                  run_time=1.5)

        # ── Layer loop ──
        for layer in range(3):
            alpha = layer / 2.0

            # QKV labels
            q_label = Text("Q", font_size=22, color=EMERALD).next_to(encoder_bars, UP, buff=0.3).shift(LEFT * 3.5)
            k_label = Text("K", font_size=22, color=RED_LINE).next_to(q_label, RIGHT, buff=2.5)
            v_label = Text("V", font_size=22, color=TEAL).next_to(k_label, RIGHT, buff=2.5)

            # Attention grid
            w = attention_weight_12(h=0.7 + 0.3 * layer)
            att_grid = make_att_grid(w, sz=0.24)
            att_grid.move_to(encoder_bars.get_center() + DOWN * 1.8)
            att_box = SurroundingRectangle(att_grid, color=GRAY, buff=0.08,
                                           stroke_width=0.8)

            # Multi-head: three small grids
            w2 = attention_weight_12(h=0.9)
            w3 = attention_weight_12(h=1.1)
            heads = VGroup()
            for idx, ww in enumerate([w, w2, w3]):
                hg = make_att_grid(ww, sz=0.1)
                hg.scale(0.5)
                heads.add(hg)
            heads.arrange(RIGHT, buff=0.4)
            head_label = Text("Multi-head Attention", font_size=18, color=GRAY)
            head_label.next_to(heads, DOWN, buff=0.15)
            heads_group = VGroup(heads, head_label)
            heads_group.move_to(encoder_bars.get_center() + DOWN * 2.6)

            # Refined bars after FFN
            refined = VGroup()
            for i, b in enumerate(encoder_bars):
                shade = interpolate_color(PURPLE, TEAL, 0.3 + 0.5 * alpha)
                nb = Rectangle(width=b.width, height=b.height * (0.8 + 0.4 * np.random.rand()),
                               fill_color=shade, fill_opacity=0.85,
                               stroke_color=WHITE, stroke_width=0.5)
                nb.move_to(b.get_center())
                refined.add(nb)

            # Short pulse to show "attention happening"
            att_glow = att_grid.copy().set_stroke(GOLD, width=1).shift(RIGHT * 8)
            glow = Dot(encoder_bars.get_center() + [0, -2, 0], radius=6,
                       color=GOLD, fill_opacity=0.1).scale(0.5)
            glow_anim = FadeIn(glow, scale=0.3)

            if layer == 0:
                self.play(Write(q_label), Write(k_label), Write(v_label),
                          run_time=1)
                self.play(
                    LaggedStart(*[cell.animate.set_stroke(GOLD, width=0.8)
                                  for cell in att_grid[::12]],
                                lag_ratio=0.02),
                    Create(att_box), run_time=2.5
                )
                self.play(FadeIn(heads_group), run_time=1.8)
                self.wait(0.5)

            elif layer == 1:
                self.play(
                    *[t.animate.set_opacity(0.6) for t in (q_label, k_label, v_label)],
                    Transform(att_grid, make_att_grid(attention_weight_12(h=1.2), sz=0.24).move_to(att_grid.get_center())),
                    Transform(heads_group[0], VGroup(*[
                        make_att_grid(attention_weight_12(h=1.0 + 0.3 * i), sz=0.1).scale(0.5)
                        for i in range(3)]).arrange(RIGHT, buff=0.4).move_to(heads_group[0].get_center())),
                    run_time=2
                )
                self.wait(0.3)

            else:
                # layer == 2 — quicker, show abstraction
                self.play(
                    *[t.animate.set_opacity(0.4) for t in (q_label, k_label, v_label)],
                    att_grid.animate.set_opacity(0.7),
                    refined.animate.set_opacity(0.9),
                    run_time=1.5
                )
                self.wait(0.3)

            # FFN: transform to refined
            self.play(
                LaggedStart(*[Transform(b, nb)
                              for b, nb in zip(encoder_bars, refined)],
                            lag_ratio=0.03),
                run_time=2 if layer == 0 else 1.2
            )
            encoder_bars = refined
            self.wait(0.5)

        # Clean up encoder artifacts
        self.play(
            FadeOut(q_label), FadeOut(k_label), FadeOut(v_label),
            FadeOut(att_grid), FadeOut(att_box),
            FadeOut(heads_group),
            run_time=1
        )

        # ════════════════════════════════════════════════
        #  STAGE 5 — LATENT BOTTLENECK
        # ════════════════════════════════════════════════
        t5 = Text("Latent Bottleneck", font_size=32).to_edge(UP).shift(DOWN * 0.2)

        latent_vals = [patch_data(i)[1] for i in range(4)]
        latent_bars = build_bars(latent_vals, 4, PURPLE, width=0.5, max_h=1.3,
                                 center=encoder_bars.get_center())

        # Animate compression: fade encoder bars, grow latent bars from center
        self.play(Transform(t1, t5), run_time=0.8)
        self.play(
            FadeOut(encoder_bars, scale=0.3),
            LaggedStart(*[GrowFromCenter(lb) for lb in latent_bars],
                        lag_ratio=0.15),
            run_time=2.5
        )
        glow_latent = Dot(latent_bars.get_center(), radius=2, color=PURPLE,
                          fill_opacity=0.15)
        self.play(FadeIn(glow_latent, scale=0.5), run_time=0.8)
        self.wait(1)

        # ════════════════════════════════════════════════
        #  STAGE 6 — REDSHIFT REGRESSION HEAD
        # ════════════════════════════════════════════════
        t6 = Text("Redshift Regression Head", font_size=30).to_edge(UP).shift(DOWN * 0.2)

        branch = Arrow(latent_bars.get_top() + [0.5, 0, 0],
                       latent_bars.get_top() + [3.5, 1.8, 0],
                       color=GOLD, stroke_width=2.5, buff=0.1)
        z_text = Text("z = 0.84", font_size=36, color=GOLD)
        z_text.next_to(branch.get_end(), RIGHT, buff=0.3)
        z_box = SurroundingRectangle(z_text, color=GOLD, buff=0.15, stroke_width=1.5)

        self.play(Transform(t1, t6), GrowArrow(branch), run_time=1.5)
        self.play(Write(z_text), Create(z_box), run_time=1.5)
        self.wait(1.5)

        # ════════════════════════════════════════════════
        #  STAGE 7 — DECODER
        # ════════════════════════════════════════════════
        t7 = Text("Decoder: Reconstruction Pathway", font_size=30).to_edge(UP).shift(DOWN * 0.2)

        decoder_vals = [patch_data(i)[1] for i in range(N_PATCH)]
        decoder_bars = build_bars(decoder_vals, N_PATCH, TEAL, width=0.3, max_h=1.0,
                                  center=latent_bars.get_center())

        self.play(
            Transform(t1, t7),
            FadeOut(branch), FadeOut(z_text), FadeOut(z_box),
            FadeOut(glow_latent),
            run_time=1
        )
        self.play(
            FadeOut(latent_bars, scale=0.3),
            LaggedStart(*[GrowFromCenter(db) for db in decoder_bars],
                        lag_ratio=0.06),
            run_time=2.5
        )
        self.wait(0.5)

        # Decoder attention (quick, mirrored)
        dec_att_w = attention_weight_12(h=0.8)
        dec_att = make_att_grid(dec_att_w, sz=0.24)
        dec_att.move_to(decoder_bars.get_center() + DOWN * 1.8)
        dec_box = SurroundingRectangle(dec_att, color=GRAY, buff=0.08, stroke_width=0.8)

        self.play(Create(dec_box), FadeIn(dec_att, shift=UP), run_time=1.5)
        self.wait(0.5)
        self.play(FadeOut(dec_att), FadeOut(dec_box), run_time=0.8)

        # ════════════════════════════════════════════════
        #  STAGE 8 — OUTPUT / RECONSTRUCTION
        # ════════════════════════════════════════════════
        t8 = Text("Denoised Reconstruction", font_size=32).to_edge(UP).shift(DOWN * 0.2)

        # Fade decoder bars, reveal reconstructed spectrum on axes
        clean_line = build_spectrum_line(lam, clean, PURPLE, 2.5)
        clean_line.set_opacity(0)
        original_fade = raw_line.copy().set_opacity(0.25)

        self.play(Transform(t1, t8), run_time=0.8)
        self.play(
            FadeOut(decoder_bars, scale=0.3),
            FadeIn(clean_line, shift=UP * 0.5),
            FadeIn(original_fade),
            run_time=2.5
        )

        # Animate noise being removed (clean line pulses, noise fades)
        self.play(
            clean_line.animate.set_stroke(width=3),
            original_fade.animate.set_opacity(0.12),
            run_time=1.5
        )

        # Flash effect to highlight
        highlight_rect = Rectangle(
            width=14, height=2.5,
            stroke_color=GOLD, stroke_width=3, fill_opacity=0
        ).move_to(axes.get_center())
        self.play(Create(highlight_rect), run_time=0.5)
        self.wait(0.5)
        self.play(FadeOut(highlight_rect), run_time=0.5)

        # Final labels on spectrum: emission lines still visible
        final_labels = VGroup()
        for nm, wl in label_data:
            idx = np.argmin(np.abs(lam - wl))
            lbl = Text(nm, font_size=16, color=PURPLE).next_to(
                axes.c2p(wl, clean[idx]), UP, buff=0.1)
            final_labels.add(lbl)
        self.play(Write(final_labels, lag_ratio=0.15), run_time=1.5)
        self.wait(2)

        # Fade out everything
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)
        self.wait(0.5)
