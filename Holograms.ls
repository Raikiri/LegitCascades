[declaration: "config"]
{{
  #define POS_FIRST_LAYOUT 1
  const float pi = 3.141592f;
}}

[blendmode: alphablend]
void OverlayTexShader(
  sampler2D tex,
  out vec4 color)
{{
  uvec2 pixel_idx = uvec2(gl_FragCoord.xy);
  uvec2 tex_size = uvec2(textureSize(tex, 0));
  if(pixel_idx.x < tex_size.x && pixel_idx.y < tex_size.y)
  {
    color = vec4(texelFetch(tex, ivec2(pixel_idx), 0).rgb, 1.0);
  }else
  {
    color = vec4(0.0);
  }
}}


[include:
  "atlas_layout",
  "probe_atlas",
  "raymarching",
  "probe_regular_grid",
  "bilinear_interpolation",
  "merging"]
[declaration: "rc_probe_casting"]
{{
  vec4 CastMergedIntervalBilinearFix(
    uvec2 size,
    vec2 screen_pos,
    vec2 dir,
    vec2 interval_minmax,
    float step_size,
    CascadeLayout prev_cascade_layout,
    ProbeLayout prev_probe_layout,
    GridTransform prev_probe_to_screen,
    uint prev_cascade_idx,
    uint prev_dir_idx,
    uint cascades_count,
    sampler2D prev_atlas_tex,
    sampler2D scene_tex)
  {
    if(prev_probe_layout.count.x == 0u || prev_probe_layout.count.y == 0u ||
       prev_probe_layout.size.x == 0u || prev_probe_layout.size.y == 0u)
      return vec4(0.0f, 0.0f, 0.0f, 1.0f);
    GridTransform screen_to_prev_probe = GetInverseTransform(prev_probe_to_screen);
    vec2 prev_probe_idx = ApplyTransform(screen_to_prev_probe, screen_pos);

    BilinearSamples bilinear_samples = GetBilinearSamples(prev_probe_idx);
    vec4 weights = GetBilinearWeights(bilinear_samples.ratio);

    vec4 merged_interval = vec4(0.0f);
    for(uint i = 0u; i < 4u; i++)
    {
      uvec2 prev_probe_idx = uvec2(clamp(bilinear_samples.base_idx + GetBilinearOffset(i), ivec2(0), ivec2(prev_probe_layout.count) - ivec2(1)));
      uvec2 cascade_texel = GetCascadeTexel(prev_probe_idx, prev_dir_idx, prev_cascade_layout.size, prev_probe_layout.size);
      uvec2 atlas_texel = prev_cascade_layout.offset + cascade_texel;
      vec4 prev_interval = prev_cascade_idx < cascades_count ? texelFetch(prev_atlas_tex, ivec2(atlas_texel), 0) : vec4(0.0);

      vec2 prev_probe_screen_pos = ApplyTransform(prev_probe_to_screen, vec2(prev_probe_idx));

      vec2 ray_start = screen_pos + dir * interval_minmax.x;
      vec2 ray_end = prev_probe_screen_pos + dir * interval_minmax.y;                
      vec4 hit_radiance = RaymarchRay(size, ray_start, ray_end, step_size, scene_tex);
      merged_interval += MergeIntervals(hit_radiance, prev_interval) * weights[i];
    }
    return merged_interval;
  }

  vec4 InterpProbe(
    vec2 screen_pos,
    uint dir_idx,
    uvec2 probe_count,
    CascadeLayout cascade_layout,
    ProbeLayout probe_layout,
    GridTransform prev_probe_to_screen,
    sampler2D atlas_tex)
  {
    GridTransform screen_to_prev_probe = GetInverseTransform(prev_probe_to_screen);
    vec2 probe_idx2f = ApplyTransform(screen_to_prev_probe, screen_pos);

    BilinearSamples bilinear_samples = GetBilinearSamples(probe_idx2f);
    vec4 weights = GetBilinearWeights(bilinear_samples.ratio);

    vec4 interp_interval = vec4(0.0f);
    for(uint i = 0u; i < 4u; i++)
    {
      uvec2 probe_idx = uvec2(clamp(bilinear_samples.base_idx + GetBilinearOffset(i), ivec2(0), ivec2(probe_layout.count) - ivec2(1)));
      uvec2 cascade_texel = GetCascadeTexel(probe_idx, dir_idx, cascade_layout.size, probe_layout.size);
      uvec2 atlas_texel = cascade_layout.offset + cascade_texel;
      vec4 interval = texelFetch(atlas_tex, ivec2(atlas_texel), 0);
      interp_interval += interval * weights[i];
    }
    return interp_interval;
  }
  float GetIntervalStart(uint cascade_idx, float interval_scaling)
  {
    return pow(interval_scaling, float(cascade_idx)) - 1.0f;
  }
  vec2 GetIntervalMinmax(uint cascade_idx, float interval_scaling)
  {
    return vec2(GetIntervalStart(cascade_idx, interval_scaling), GetIntervalStart(cascade_idx + 1u, interval_scaling));
  }
}}

[include: "rc_probe_casting", "pcg"]
void RaymarchAtlasShader(
  uvec2 size,
  uvec2 c0_size,
  float c0_dist,
  float cm1_mult,
  int cascade_scaling_pow2,
  uint cascades_count,
  uint dir_scaling,
  uvec2 c0_probe_size,
  sampler2D scene_tex,
  sampler2D prev_atlas_tex,
  out vec4 color)
{{
  uvec2 atlas_texel_idx = uvec2(gl_FragCoord.xy);

  uint c0_dirs_count = uint(c0_probe_size.x * c0_probe_size.y);
  uvec2 atlas_size = GetAtlasSize(cascade_scaling_pow2, uint(cascades_count), uvec2(c0_size));

  AtlasTexelLocation loc = GetAtlasPixelLocationPosFirst(
    atlas_texel_idx,
    cascade_scaling_pow2,
    uvec2(c0_probe_size),
    uint(dir_scaling),
    uint(cascades_count),
    uvec2(c0_size));
  vec2 c0_probe_spacing = GetC0ProbeSpacing(size, loc.c0_probe_layout.count);
  uint prev_cascade_idx = loc.cascade_idx + 1u;
  CascadeLayout prev_cascade_layout = GetCascadeLayout(cascade_scaling_pow2, prev_cascade_idx, c0_size);

  ProbeLayout prev_probe_layout = GetProbeLayout(
    prev_cascade_idx, prev_cascade_layout.size, c0_probe_size, loc.probe_scaling);

  //vec2 prev_probe_spacing = GetProbeSpacing(c0_probe_spacing, prev_cascade_idx, loc.probe_scaling.size_scaling);
  vec2 prev_probe_spacing = GetProbeUniformSpacing(size, prev_probe_layout.count);
  GridTransform prev_probe_to_screen = GetProbeToScreenTransform(prev_probe_spacing);
  uvec2 dir_idx2 = GetProbeDirIdx2(loc.dir_idx, loc.probe_layout.size);
  if(
    loc.cascade_idx < cascades_count &&
    loc.probe_idx.x < loc.probe_layout.count.x &&
    loc.probe_idx.y < loc.probe_layout.count.y &&
    loc.dir_idx < loc.probe_layout.dirs_count)
  {
    vec2 probe_spacing = GetProbeUniformSpacing(size, loc.probe_layout.count);
    //vec2 probe_spacing = GetProbeSpacing(c0_probe_spacing, loc.cascade_idx, loc.probe_scaling.size_scaling);
    GridTransform probe_to_screen = GetProbeToScreenTransform(probe_spacing);
    vec2 screen_pos = ApplyTransform(probe_to_screen, vec2(loc.probe_idx));

    for(uint dir_number = 0u; dir_number < dir_scaling; dir_number++)
    {
      uint prev_dir_idx = loc.dir_idx * dir_scaling + dir_number;
      float ang = 2.0f * pi * (float(prev_dir_idx) + 0.5f) / float(prev_probe_layout.dirs_count);
      vec2 ray_dir = vec2(cos(ang), sin(ang));
      float step_size = max(1e-4f, 2.0f * pow(loc.probe_scaling.spacing_scaling, float(loc.cascade_idx)));
      float cm1_dist = c0_probe_spacing.x * cm1_mult;
      vec4 radiance = CastMergedIntervalBilinearFix(
        size,
        screen_pos,
        ray_dir,
        GetIntervalMinmax(loc.cascade_idx, float(dir_scaling)) * c0_dist + vec2(cm1_dist),
        step_size,
        prev_cascade_layout,
        prev_probe_layout,
        prev_probe_to_screen,
        prev_cascade_idx,
        prev_dir_idx,
        cascades_count,
        prev_atlas_tex,
        scene_tex);

      color += radiance / float(dir_scaling);
    }
  }else
  {
    if(atlas_texel_idx.x < atlas_size.x && atlas_texel_idx.y < atlas_size.y)
    {
      color = vec4(0.1, 0.0, 0.0, 1.0);
    }else
    {
      color = vec4(0.0, 0.0, 0.0, 1.0);
    }
  }
}}

[include: "rc_probe_casting", "pcg"]
void FinalGatheringShader(
  uvec2 size,
  uvec2 c0_size,
  float cm1_mult,
  int cascade_scaling_pow2,
  uint cascades_count,
  uint dir_scaling,
  uvec2 c0_probe_size,
  sampler2D scene_tex,
  sampler2D atlas_tex,
  out vec4 color)
{{
  vec2 screen_pos = gl_FragCoord.xy;
  uint cascade_idx = 0u;

  ProbeLayout c0_probe_layout;
  c0_probe_layout = GetC0ProbeLayout(c0_size, c0_probe_size);
  ProbeScaling probe_scaling = GetProbeScaling(cascade_scaling_pow2, dir_scaling);
  CascadeLayout cascade_layout = GetCascadeLayout(cascade_scaling_pow2, cascade_idx, c0_size);
  vec2 c0_probe_spacing = GetC0ProbeSpacing(size, c0_probe_layout.count);
  ProbeLayout probe_layout = GetProbeLayout(cascade_idx, cascade_layout.size, c0_probe_size, probe_scaling);
  //vec2 probe_spacing = GetProbeSpacing(c0_probe_spacing, cascade_idx, probe_scaling.size_scaling);
  vec2 probe_spacing = GetProbeUniformSpacing(size, probe_layout.count);
  GridTransform probe_to_screen = GetProbeToScreenTransform(probe_spacing);
  uint dirs_count = probe_layout.size.x * probe_layout.size.y;

  vec4 fluence = vec4(0.0f);
  for(uint dir_idx = 0u; dir_idx < dirs_count; dir_idx++)
  {
    /*vec4 radiance = InterpProbe(
      screen_pos,
      dir_idx,
      probe_layout.count,
      cascade_layout,
      probe_layout,
      probe_to_screen,
      atlas_tex);*/
    float ang = (float(dir_idx) + 0.5f) / float(dirs_count) * 2.0f * pi;
    vec4 radiance = CastMergedIntervalBilinearFix(
      size,
      screen_pos,
      vec2(cos(ang), sin(ang)),
      vec2(0.0f, c0_probe_spacing.x * cm1_mult),
      2.0f,
      cascade_layout,
      probe_layout,
      probe_to_screen,
      cascade_idx,
      dir_idx,
      cascades_count,
      atlas_tex,
      scene_tex);
  
    fluence += radiance / float(dirs_count);
  }

  color = fluence;
}}

[rendergraph]
[include: "fps", "atlas_layout", "bessel"]
void RenderGraphMain()
{{
  int run = SliderInt("Run", 0, 1);
  if(run == 1)
  {
    uvec2 size = GetSwapchainImage().GetSize();
    ClearShader(GetSwapchainImage());
    uvec2 c0_size;
    c0_size.x = SliderInt("c0_size.x", 1, 1024, 256);
    c0_size.y = size.y * c0_size.x / size.x;
    int cascade_scaling_pow2 = SliderInt("cascade_scaling_pow2", -1, 1, 0);
    uint cascades_count = SliderInt("cascades count", 1, 10, 3);
    uint dir_scaling = SliderInt("dir_scaling", 1, 10, 4);
    uvec2 c0_probe_size = uvec2(SliderInt("c0_probe_size", 1, 10, 2));
    float c0_dist = SliderFloat("c0_dist", 0.0f, 40.0f, 10.0f);
    float cm1_mult = SliderFloat("cm1_mult", 0.0f, 5.0f, 1.0f);

    Image scene_img = GetImage(size, rgba16f);
    SceneShader(size, scene_img);

    uvec2 atlas_size = GetAtlasSize(cascade_scaling_pow2, cascades_count, c0_size);
    Text("c0_size: " + c0_size + " atlas size: " + atlas_size);
    Image merged_atlas_img = GetImage(atlas_size, rgba16f);
    Image prev_atlas_img = GetImage(atlas_size, rgba16f);
    RaymarchAtlasShader(
      size,
      c0_size,
      c0_dist,
      cm1_mult,
      cascade_scaling_pow2,
      cascades_count,
      dir_scaling,
      c0_probe_size,
      scene_img,
      prev_atlas_img,
      merged_atlas_img
    );
    CopyShader(merged_atlas_img, prev_atlas_img);

    FinalGatheringShader(
      size,
      c0_size,
      cm1_mult,
      cascade_scaling_pow2,
      cascades_count,
      dir_scaling,
      c0_probe_size,
      scene_img,
      merged_atlas_img,
      GetSwapchainImage()
    );
    OverlayTexShader(
      merged_atlas_img,
      GetSwapchainImage());
  }

  Text("Fps: " + GetSmoothFps());
}}

void ClearShader(out vec4 col)
{{
  col = vec4(0.0f, 0.0f, 0.0f, 1.0f);
}}

void CopyShader(sampler2D tex, out vec4 col)
{{
  col = texelFetch(tex, ivec2(gl_FragCoord.xy), 0);
}}
[declaration: "atlas_layout"]
{{
  uvec2 GetAtlasSize(int cascade_scaling_pow2, uint cascades_count, uvec2 c0_size)
  {
    if(cascade_scaling_pow2 == 0) //constant size
    {
      return uvec2(c0_size.x, c0_size.y * cascades_count);
    }else
    if(cascade_scaling_pow2 == -1) //cascades get 2x smaller
    {
      return uvec2(c0_size.x, c0_size.y * uint(2));
    }else //cascades get 2x larger
    {
      return uvec2(c0_size.x, c0_size.y * ((uint(1) << cascades_count) - uint(1)));
    }
  }
}}

[declaration: "grid_transform"]
{{
  struct GridTransform
  {
    vec2 spacing;
    vec2 origin;
  };
  GridTransform GetGridTransform(vec2 src_grid_spacing, vec2 src_grid_origin)
  {
    return GridTransform(src_grid_spacing, src_grid_origin);
  }
  GridTransform GetInverseTransform(GridTransform transform)
  {
    vec2 inv_spacing = vec2(1.0) / max(vec2(1e-7f), transform.spacing);
    return GridTransform(inv_spacing, -transform.origin * inv_spacing);
  }
  GridTransform CombineTransform(GridTransform src_to_tmp, GridTransform tmp_to_dst)
  {
    return GridTransform(src_to_tmp.spacing * tmp_to_dst.spacing, src_to_tmp.origin + tmp_to_dst.spacing + tmp_to_dst.origin);
  }
  GridTransform GetSrcToDstTransform(GridTransform src_transform, GridTransform dst_transform)
  {
    return CombineTransform(src_transform, GetInverseTransform(dst_transform));
  }
  vec2 ApplyTransform(GridTransform transform, vec2 p)
  {
    return transform.spacing * p + transform.origin;
  }
}}
[declaration: "probe_regular_grid"]
[include: "grid_transform"]
{{
  vec2 GetC0ProbeSpacing(uvec2 size, uvec2 c0_probes_count)
  {
    return vec2(size) / max(vec2(1e-5f), vec2(c0_probes_count));
  }
  vec2 GetProbeSpacing(vec2 c0_probe_spacing, uint cascade_idx, float probe_size_scaling)
  {
    return c0_probe_spacing * pow(vec2(probe_size_scaling), vec2(cascade_idx));
  }
  vec2 GetProbeUniformSpacing(uvec2 size, uvec2 probes_count)
  {
    return vec2(size) / vec2(max(uvec2(1), probes_count));
  }  
  GridTransform GetProbeToScreenTransform(vec2 probe_spacing)
  {
    return GridTransform(probe_spacing, probe_spacing * 0.5f);
  }
}}

[include: "config"]
[declaration: "probe_atlas"]
{{
  struct CascadeLayout
  {
    uvec2 size;
    uvec2 offset;
  };
  CascadeLayout GetCascadeLayout(int cascade_scaling_pow2, uint cascade_idx, uvec2 c0_size)
  {
    CascadeLayout cascade_layout;
    if(cascade_scaling_pow2 == 0) //constant size
    {
      cascade_layout.size = c0_size;
      cascade_layout.offset = uvec2(0u, c0_size.y * cascade_idx);
    }else
    if(cascade_scaling_pow2 == -1) //cascades get 2x smaller
    {
      cascade_layout.size = uvec2(c0_size.x, c0_size.y >> cascade_idx);
      cascade_layout.offset = uvec2(0u, (c0_size.y - (c0_size.y >> cascade_idx)) * 2u);
    }else //cascades get 2x larger
    {
      cascade_layout.size = uvec2(c0_size.x, c0_size.y << cascade_idx);
      cascade_layout.offset = uvec2(0u, c0_size.y * ((1u << cascade_idx) - 1u));
    }
    return cascade_layout;
  }

  struct ProbeScaling
  {
    uint dirs_scaling;
    float spacing_scaling;
    vec2 size_scaling;
  };

  ProbeScaling GetProbeScaling(
    int cascade_scaling_pow2,
    uint dirs_scaling)
  {
    ProbeScaling probe_scaling;
    vec2 aspect = vec2(1.0f, pow(2.0f, float(cascade_scaling_pow2)));
    probe_scaling.spacing_scaling = 1.0 * sqrt(float(dirs_scaling) / (aspect.x * aspect.y));
    probe_scaling.size_scaling = probe_scaling.spacing_scaling * aspect;
    probe_scaling.dirs_scaling = dirs_scaling;
    return probe_scaling;
  }
  struct ProbeLayout
  {
    uvec2 count;
    uvec2 size;
    uint dirs_count;
  };
  ProbeLayout GetC0ProbeLayout(uvec2 c0_size, uvec2 c0_probe_size)
  {
    ProbeLayout probe_layout;
    probe_layout.size = c0_probe_size;
    probe_layout.count = c0_size / max(uvec2(1u), probe_layout.size);
    return probe_layout;
  }
  ProbeLayout GetProbeLayout(
    uint cascade_idx,
    uvec2 cascade_size,
    uvec2 c0_probe_size,
    ProbeScaling probe_scaling)
  {
    ProbeLayout probe_layout;
    vec2 probe_scale = pow(probe_scaling.size_scaling, vec2(cascade_idx));
    vec2 probe_size_2f = max(vec2(1.0f), ceil(vec2(c0_probe_size) * probe_scale - vec2(1e-5f)));
    probe_layout.size = uvec2(probe_size_2f);
    probe_layout.dirs_count = c0_probe_size.x * c0_probe_size.y * uint(1e-5f + pow(float(probe_scaling.dirs_scaling), float(cascade_idx)));
    if(probe_layout.size.x * probe_layout.size.y < probe_layout.dirs_count)
    {
      probe_layout.size.x++;
    }
    if(probe_layout.size.x * probe_layout.size.y < probe_layout.dirs_count)
    {
      probe_layout.size.y++;
    }
    probe_layout.count = cascade_size / probe_layout.size;
    return probe_layout;
  }

  uvec2 GetProbeDirIdx2(uint dir_idx, uvec2 probe_size)
  {
    uint s = max(1u, probe_size.x);
    return uvec2(dir_idx % s, dir_idx / s);
  }


  struct AtlasTexelLocation
  {
    ProbeLayout c0_probe_layout;
    CascadeLayout cascade_layout;
    uint cascade_idx;
    uvec2 probe_idx;
    uint dir_idx;

    ProbeScaling probe_scaling;
    ProbeLayout probe_layout;
  };

  AtlasTexelLocation GetAtlasTexelLocationDirFirst(uvec2 atlas_texel, uint cascades_count)
  {
    AtlasTexelLocation loc;

    return loc;
  }
  AtlasTexelLocation GetAtlasPixelLocationPosFirst(
    uvec2 atlas_texel,
    int cascade_scaling_pow2,
    uvec2 c0_probe_size,
    uint dir_scaling,
    uint cascades_count,
    uvec2 c0_size)
  {
    AtlasTexelLocation loc;
    for(loc.cascade_idx = 0u; loc.cascade_idx < cascades_count; loc.cascade_idx++)
    {
      loc.cascade_layout = GetCascadeLayout(cascade_scaling_pow2, loc.cascade_idx, c0_size);
      if(atlas_texel.y >= loc.cascade_layout.offset.y && atlas_texel.y < loc.cascade_layout.offset.y + loc.cascade_layout.size.y)
      {
        break;
      }
    }

    loc.c0_probe_layout = GetC0ProbeLayout(c0_size, c0_probe_size);

    loc.probe_scaling = GetProbeScaling(cascade_scaling_pow2, dir_scaling);
    loc.probe_layout = GetProbeLayout(loc.cascade_idx, loc.cascade_layout.size, c0_probe_size, loc.probe_scaling);
    
    uvec2 cascade_texel = atlas_texel - loc.cascade_layout.offset;

    #if(POS_FIRST_LAYOUT)
      uvec2 cells_count = loc.probe_layout.size;
      uvec2 cell_size = loc.cascade_layout.size / max(uvec2(1), cells_count);
      if(cells_count.x > 0u && cells_count.y > 0u && cell_size.x > 0u && cell_size.y > 0u)
      {
        loc.probe_idx = cascade_texel % max(uvec2(1u), cell_size);
        uvec2 dir_idx2 = cascade_texel / max(uvec2(1), cell_size);
        if(dir_idx2.x < loc.probe_layout.size.x && dir_idx2.y < loc.probe_layout.size.y)
          loc.dir_idx = dir_idx2.x + dir_idx2.y * loc.probe_layout.size.x;
        else
          loc.dir_idx = loc.probe_layout.size.x * loc.probe_layout.size.y;
      }else
      {
        loc.dir_idx = loc.probe_layout.size.x * loc.probe_layout.size.y;
        loc.probe_idx = loc.probe_layout.count;
      }
    #else
      loc.probe_idx = cascade_texel / max(uvec2(1u), loc.probe_layout.size);
      uvec2 dir_idx2 = cascade_texel % max(uvec2(1u), loc.probe_layout.size);
      loc.dir_idx = dir_idx2.x + dir_idx2.y * loc.probe_layout.size.x;
    #endif

    return loc;
  }

  uvec2 GetCascadeTexel(uvec2 probe_idx, uint dir_idx, uvec2 cascade_size, uvec2 probe_size)
  {
    #if(POS_FIRST_LAYOUT)
      uvec2 cells_count = probe_size;
      uvec2 cell_size = cascade_size / max(uvec2(1u), cells_count);
      uvec2 cell_idx = GetProbeDirIdx2(dir_idx, probe_size);
      return cell_size * cell_idx + probe_idx;
    #else
      uvec2 dir_idx2 = GetProbeDirIdx2(dir_idx, probe_size);
      return probe_idx * probe_size + dir_idx2;
    #endif
  }
}}


[declaration: "raymarching"]
{{
  vec4 RaymarchRay(uvec2 size, vec2 ray_start, vec2 ray_end, float step_size, sampler2D scene_tex)
  {
    vec2 inv_size = vec2(1.0f) / vec2(size);

    vec2 delta = ray_end - ray_start;
    float len = length(delta);
    vec2 ray_dir = delta / max(1e-5f, len);

    vec3 radiance = vec3(0.0f);
    float transmittance = 1.0f;
    for(float offset = 0.0f; offset < len; offset += step_size)
    {
      vec2 ray_pos = ray_start + ray_dir * offset;
      vec2 uv_pos = ray_pos * inv_size;
      vec4 ray_sample = textureLod(scene_tex, uv_pos, 0.0);
      radiance += ray_sample.rgb * transmittance * ray_sample.a;
      transmittance *= (1.0f - ray_sample.a);
    }
    return vec4(radiance, transmittance);
  }
}}

[declaration: "scene"]
{{
  vec4 Circle(vec4 prev_radiance, vec2 delta, float radius, vec4 circle_radiance)
  {
    return length(delta) < radius ? circle_radiance : prev_radiance;
  }
}}

[include: "scene"]
void SceneShader(uvec2 size, out vec4 radiance)
{{
  radiance = vec4(0.0f);
  radiance = Circle(radiance, vec2(size / 2u) - gl_FragCoord.xy, 30.0, vec4(1.0f, 0.5f, 0.0f, 1.0f));
  //radiance = Circle(radiance, vec2(size / 2u) + vec2(10.0, 0.0) - gl_FragCoord.xy, 30.0, vec4(0.1f, 0.1f, 0.1f, 1.0f));
  radiance = Circle(radiance, vec2(size / 2u) + vec2(150.0, 50.0) - gl_FragCoord.xy, 30.0, vec4(0.0f, 0.0f, 0.0f, 1.0f));
}}

[declaration: "merging"]
{{
  vec4 MergeIntervals(vec4 near_interval, vec4 far_interval)
  {
      //return near_interval + far_interval;
      return vec4(near_interval.rgb + near_interval.a * far_interval.rgb, near_interval.a * far_interval.a);
  }
}}

[declaration: "pcg"]
{{
  //http://www.jcgt.org/published/0009/03/02/paper.pdf
  uvec3 hash33UintPcg(uvec3 v)
  {
      v = v * 1664525u + 1013904223u;
      v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
      //v += v.yzx * v.zxy; //swizzled notation is not exactly the same because components depend on each other, but works too

      v ^= v >> 16u;
      v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
      //v += v.yzx * v.zxy;
      return v;
  }

  vec3 hash3i3f(ivec3 seed)
  {
      uvec3 hash_uvec3 = hash33UintPcg(uvec3(seed));
      return vec3(hash_uvec3) * (1.0f / float(~0u));
  }
}}

[declaration: "bilinear_interpolation"]
{{
  struct BilinearSamples
  {
      ivec2 base_idx;
      vec2 ratio;
  };

  vec4 GetBilinearWeights(vec2 ratio)
  {
      return vec4(
          (1.0f - ratio.x) * (1.0f - ratio.y),
          ratio.x * (1.0f - ratio.y),
          (1.0f - ratio.x) * ratio.y,
          ratio.x * ratio.y);
  }

  ivec2 GetBilinearOffset(uint offset_index)
  {
      ivec2 offsets[4] = ivec2[4](ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(1, 1));
      return offsets[offset_index];
  }
  BilinearSamples GetBilinearSamples(vec2 pixel_idx2f)
  {
      BilinearSamples samples;
      samples.base_idx = ivec2(floor(pixel_idx2f));
      samples.ratio = fract(pixel_idx2f);
      return samples;
  }
}}
[declaration: "smoothing"]
{{
  float SmoothOverTime(float val, string name, float ratio = 0.95)
  {
    ContextVec2(name) = ContextVec2(name) * ratio + vec2(val, 1) * (1.0 - ratio);
    return ContextVec2(name).x / (1e-7f + ContextVec2(name).y);
  }
}}

[declaration: "bessel"]
{{
  //https://www.shadertoy.com/view/Wt3czM
  // License: CC0 (https://creativecommons.org/publicdomain/zero/1.0/)

  /*
      Approximations for the Bessel functions J0 and J1 and the Struve functions H0 and H1.
      https://en.wikipedia.org/wiki/Bessel_function
      https://en.wikipedia.org/wiki/Struve_function
  */

  // https://link.springer.com/article/10.1007/s40314-020-01238-z
  float BesselJ0(float x)
  {
      float xx = x * x;
      float lamb = 0.865;
      float q    = 0.7172491568;
      float p0   = 0.6312725339;
      float ps0  = 0.4308049446;
      float p1   = 0.3500347951;
      float ps1  = 0.4678202347;
      float p2   =-0.06207747907;
      float ps2  = 0.04253832927;

      float lamb4 = (lamb * lamb) * (lamb * lamb);
      float t0 = sqrt(1.0 + lamb4 * xx);
      float t1 = sqrt(t0);
      
      return xx == 0.0 ? 1.0 : 1.0/(t1 * (1.0 + q * xx)) * ((p0 + p1*xx + p2*t0) * cos(x) + ((ps0 + ps1*xx) * t0 + ps2*xx) * (sin(x)/x));
  }

  // https://www.sciencedirect.com/science/article/pii/S2211379718300111
  float BesselJ1(float x)
  {
      float xx = x * x;

      return (sqrt(1.0 + 0.12138 * xx) * (46.68634 + 5.82514 * xx) * sin(x) - x * (17.83632 + 2.02948 * xx) * cos(x)) /
            ((57.70003 + 17.49211 * xx) * pow(1.0 + 0.12138 * xx, 3.0/4.0) );
  }

  // https://research.tue.nl/nl/publications/efficient-approximation-of-the-struve-functions-hn-occurring-in-the-calculation-of-sound-radiation-quantaties(c68b8858-9c9d-4ff2-bf39-e888bb638527).html
  float StruveH0(float x)
  {
      float xx = x * x;

      return BesselJ1(x) + 1.134817700  * (1.0 - cos(x))/x - 
                          1.0943193181 * (sin(x) - x * cos(x))/xx - 
                          0.5752390840 * (x * 0.8830472903 - sin(x * 0.8830472903))/xx;
  }

  // https://research.tue.nl/nl/publications/efficient-approximation-of-the-struve-functions-hn-occurring-in-the-calculation-of-sound-radiation-quantaties(c68b8858-9c9d-4ff2-bf39-e888bb638527).html
  float StruveH1(float x)
  {
      const float pi = 3.14159265359;

      float xx = x * x;

      return 2.0/pi - BesselJ0(x) + 0.0404983827 * sin(x)/x + 
                                    1.0943193181 * (1.0 - cos(x))/xx - 
                                    0.5752390840 * (1.0 - cos(x * 0.8830472903))/xx;
  }
  // https://www.shadertoy.com/view/tlf3D2
  // evaluate integer-order Bessel function of the first kind using the midpoint rule; https://doi.org/10.1002/sapm1955341298
  // see also https://doi.org/10.2307/2695765 and https://doi.org/10.1137/130932132 for more details
  float besselJ(int n, float x)
  {
    int m = 14;
      float mm = float(m), nn = float(n);
      float s = 0.0, h = 0.5 * PI/mm;
      
      for (int k = 0; k < m; k++)
      {
          float t = h * (float(k) + 0.5);
          s += (((n & 1) == 1) ? (sin(x * sin(t)) * sin(nn * t)) : (cos(x * cos(t)) * cos(nn * t)))/mm;
      }
      
      return ((n & 1) == 1) ? s : (((((n >> 1) & 1) == 1) ? -1.0 : 1.0) * s);
  }
  //https://www.astro.rug.nl/~gipsy/sub/bessel.c
  //File:         bessel.c
  //Author:       M.G.R. Vogelaar
  
   float bessj0( float x )
  /*------------------------------------------------------------*/
  /* PURPOSE: Evaluate Bessel function of first kind and order  */
  /*          0 at input x                                      */
  /*------------------------------------------------------------*/
  {
    float ax,z;
    float xx,y,ans,ans1,ans2;

    if ((ax=abs(x)) < 8.0) {
        y=x*x;
        ans1=57568490574.0+y*(-13362590354.0+y*(651619640.7
          +y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));
        ans2=57568490411.0+y*(1029532985.0+y*(9494680.718
          +y*(59272.64853+y*(267.8532712+y*1.0))));
        ans=ans1/ans2;
    } else {
        z=8.0/ax;
        y=z*z;
        xx=ax-0.785398164;
        ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
          +y*(-0.2073370639e-5+y*0.2093887211e-6)));
        ans2 = -0.1562499995e-1+y*(0.1430488765e-3
          +y*(-0.6911147651e-5+y*(0.7621095161e-6
          -y*0.934935152e-7)));
        ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
    }
    return ans;
  }



   float bessj1( float x )
  /*------------------------------------------------------------*/
  /* PURPOSE: Evaluate Bessel function of first kind and order  */
  /*          1 at input x                                      */
  /*------------------------------------------------------------*/
  {
    float ax,z;
    float xx,y,ans,ans1,ans2;

    if ((ax=abs(x)) < 8.0) {
        y=x*x;
        ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
          +y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
        ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
          +y*(99447.43394+y*(376.9991397+y*1.0))));
        ans=ans1/ans2;
    } else {
        z=8.0/ax;
        y=z*z;
        xx=ax-2.356194491;
        ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
          +y*(0.2457520174e-5+y*(-0.240337019e-6))));
        ans2=0.04687499995+y*(-0.2002690873e-3
          +y*(0.8449199096e-5+y*(-0.88228987e-6
          +y*0.105787412e-6)));
        ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
        if (x < 0.0) ans = -ans;
    }
    return ans;
  }

   float bessy0( float x )
  /*------------------------------------------------------------*/
  /* PURPOSE: Evaluate Bessel function of second kind and order */
  /*          0 at input x.                                     */
  /*------------------------------------------------------------*/
  {
    float z;
    float xx,y,ans,ans1,ans2;

    if (x < 8.0) {
        y=x*x;
        ans1 = -2957821389.0+y*(7062834065.0+y*(-512359803.6
          +y*(10879881.29+y*(-86327.92757+y*228.4622733))));
        ans2=40076544269.0+y*(745249964.8+y*(7189466.438
          +y*(47447.26470+y*(226.1030244+y*1.0))));
        ans=(ans1/ans2)+0.636619772*bessj0(x)*log(x);
    } else {
        z=8.0/x;
        y=z*z;
        xx=x-0.785398164;
        ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
          +y*(-0.2073370639e-5+y*0.2093887211e-6)));
        ans2 = -0.1562499995e-1+y*(0.1430488765e-3
          +y*(-0.6911147651e-5+y*(0.7621095161e-6
          +y*(-0.934945152e-7))));
        ans=sqrt(0.636619772/x)*(sin(xx)*ans1+z*cos(xx)*ans2);
    }
    return ans;
  }



   float bessy1( float x )
  /*------------------------------------------------------------*/
  /* PURPOSE: Evaluate Bessel function of second kind and order */
  /*          1 at input x.                                     */
  /*------------------------------------------------------------*/
  {
    float z;
    float xx,y,ans,ans1,ans2;

    if (x < 8.0) {
        y=x*x;
        ans1=x*(-0.4900604943e13+y*(0.1275274390e13
          +y*(-0.5153438139e11+y*(0.7349264551e9
          +y*(-0.4237922726e7+y*0.8511937935e4)))));
        ans2=0.2499580570e14+y*(0.4244419664e12
          +y*(0.3733650367e10+y*(0.2245904002e8
          +y*(0.1020426050e6+y*(0.3549632885e3+y)))));
        ans=(ans1/ans2)+0.636619772*(bessj1(x)*log(x)-1.0/x);
    } else {
        z=8.0/x;
        y=z*z;
        xx=x-2.356194491;
        ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
          +y*(0.2457520174e-5+y*(-0.240337019e-6))));
        ans2=0.04687499995+y*(-0.2002690873e-3
          +y*(0.8449199096e-5+y*(-0.88228987e-6
          +y*0.105787412e-6)));
        ans=sqrt(0.636619772/x)*(sin(xx)*ans1+z*cos(xx)*ans2);
    }
    return ans;
  }


}}
  
[declaration: "fps"]
[include: "smoothing"]
{{
  float GetSmoothFps()
  {
    float dt = GetTime() - ContextFloat("prev_time");
    ContextFloat("prev_time") = GetTime();

    return 1000.0 / (1e-7f + SmoothOverTime(dt, "fps_count"));
  }
}}
