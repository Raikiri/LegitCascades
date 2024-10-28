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

  // chronicl FIX (UNOPTIMIZED) START
  // This unoptimized version is about as fast as the bilinear fix. The optimized version would be about 3x
  // faster according to some early tests.

  // Assumes the ray starts within the circle and returns the guaranteed
  // single intersection point as t, where intersection = ray_origin + t * ray_direction
  float circleRayIntersection(vec2 circle_center, float circle_radius, vec2 ray_origin, vec2 ray_direction) {
    vec2 d = ray_origin - circle_center;
    float a = dot(ray_direction, ray_direction);
    float b = 2.0 * dot(ray_direction, d);
    float c = dot(d, d) - circle_radius * circle_radius;

    float discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0) {
      return -1.0;
    }

    float sqrt_discriminant = sqrt(discriminant);
    float t1 = (-b - sqrt_discriminant) / (2.0 * a);
    float t2 = (-b + sqrt_discriminant) / (2.0 * a);

    return max(t1, t2);
  }

  float GetDirIndex(vec2 direction, uint ray_count) {
      float angle = atan(direction.y, direction.x);
      float normalized_angle = angle < 0.0 ? angle + 2.0 * 3.14159 : angle;
      float ray_index = normalized_angle * float(ray_count) / (2.0 * pi) - 0.5;
      if (ray_index < 0.) {
          ray_index += float(ray_count);
      }
      return ray_index;
  }

  vec4 CastMergedIntervalAlteredBilinearFix(
    uvec2 size,
    vec2 screen_pos,
    vec2 dir,
    vec2 interval_minmax,
    float step_size,
    CascadeLayout prev_cascade_layout,
    ProbeLayout prev_probe_layout,
    GridTransform prev_probe_to_screen,
    uint prev_cascade_idx,
    // unused
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
      vec2 prev_probe_screen_pos = ApplyTransform(prev_probe_to_screen, vec2(prev_probe_idx));

      // intersecting the ray with the interval start circle of the probe from the previous cascade.
      float ray_end_t = circleRayIntersection(prev_probe_screen_pos, interval_minmax.y, screen_pos, dir);
      vec2 ray_start = screen_pos + dir * interval_minmax.x;
      vec2 ray_end = screen_pos + dir * ray_end_t;
      vec4 hit_radiance = RaymarchRay(size, ray_start, ray_end, step_size, scene_tex);

      // finding the rays/directions from the previous cascades probe that start closest to the intersection point
      vec2 prev_dir = normalize(ray_end - prev_probe_screen_pos);
      float prev_dir_idx = GetDirIndex(prev_dir, prev_probe_layout.dirs_count);
      uint dir_lower = uint(floor(prev_dir_idx)) % prev_probe_layout.dirs_count;
      uint dir_upper = uint(ceil(prev_dir_idx)) % prev_probe_layout.dirs_count;

      // getting the radiance of those rays/directions
      uvec2 cascade_texel_lower = GetCascadeTexel(prev_probe_idx, dir_lower, prev_cascade_layout.size, prev_probe_layout.size);      
      uvec2 atlas_texel_lower = prev_cascade_layout.offset + cascade_texel_lower;
      vec4 prev_interval_lower = prev_cascade_idx < cascades_count ? texelFetch(prev_atlas_tex, ivec2(atlas_texel_lower), 0) : vec4(0.0);

      uvec2 cascade_texel_upper = GetCascadeTexel(prev_probe_idx, dir_upper, prev_cascade_layout.size, prev_probe_layout.size);      
      uvec2 atlas_texel_upper = prev_cascade_layout.offset + cascade_texel_upper;
      vec4 prev_interval_upper = prev_cascade_idx < cascades_count ? texelFetch(prev_atlas_tex, ivec2(atlas_texel_upper), 0) : vec4(0.0);

      // linearly interpolating the two rays/directions
      float weight = fract(prev_dir_idx);
      vec4 prev_interval = (1. - weight) * prev_interval_lower + weight * prev_interval_upper;

      // merging ray with radiance from previous cascade probe
      merged_interval += MergeIntervals(hit_radiance, prev_interval) * weights[i];
    }
    return merged_interval;
  }
  // chronicl FIX END

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
  int cascade_scaling_pow2,
  uint cascades_count,
  uint dir_scaling,
  uvec2 c0_probe_size,
  sampler2D scene_tex,
  sampler2D prev_atlas_tex,
  uint fix_checkerboard,
  uint force_bilinear,
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

      vec4 radiance;
      // For the first two cascades we use the bilinear fix to avoid checkerboard artifacts. Could also just do normal rc here.
      if (force_bilinear > 0u || (fix_checkerboard > 0u && loc.cascade_idx < 2u)) {
        radiance = CastMergedIntervalBilinearFix(
          size,
          screen_pos,
          ray_dir,
          GetIntervalMinmax(loc.cascade_idx, float(dir_scaling)) * c0_dist,
          step_size,
          prev_cascade_layout,
          prev_probe_layout,
          prev_probe_to_screen,
          prev_cascade_idx,
          prev_dir_idx,
          cascades_count,
          prev_atlas_tex,
          scene_tex);
      } else {
        radiance = CastMergedIntervalAlteredBilinearFix(
          size,
          screen_pos,
          ray_dir,
          GetIntervalMinmax(loc.cascade_idx, float(dir_scaling)) * c0_dist,
          step_size,
          prev_cascade_layout,
          prev_probe_layout,
          prev_probe_to_screen,
          prev_cascade_idx,
          prev_dir_idx,
          cascades_count,
          prev_atlas_tex,
          scene_tex);
      }
       

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
    vec4 radiance = InterpProbe(
      screen_pos,
      dir_idx,
      probe_layout.count,
      cascade_layout,
      probe_layout,
      probe_to_screen,
      atlas_tex);
    fluence += radiance / float(dirs_count);
  }

  color = fluence;
}}

[rendergraph]
[include: "fps", "atlas_layout"]
void RenderGraphMain()
{{
  uvec2 size = GetSwapchainImage().GetSize();
  ClearShader(GetSwapchainImage());
  uvec2 c0_size;
  c0_size.x = SliderInt("c0_size.x", 1, 1024, 256);
  c0_size.y = size.y * c0_size.x / size.x;
  int cascade_scaling_pow2 = SliderInt("cascade_scaling_pow2", -1, 1, 0);
  uint cascades_count = SliderInt("cascades count", 1, 10, 4);
  uint dir_scaling = SliderInt("dir_scaling", 1, 10, 4);
  uvec2 c0_probe_size = uvec2(SliderInt("c0_probe_size", 1, 10, 2));
  float c0_dist = SliderFloat("c0_dist", 0.0f, 40.0f, 3.0f);
  uint fix_checkerboard = SliderInt("fix checkerboard", 0, 1, 1);
  uint force_bilinear = SliderInt("force bilinear fix", 0, 1, 0);

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
    cascade_scaling_pow2,
    cascades_count,
    dir_scaling,
    c0_probe_size,
    scene_img,
    prev_atlas_img,
    fix_checkerboard,
    force_bilinear,
    merged_atlas_img
  );
  CopyShader(merged_atlas_img, prev_atlas_img);

  FinalGatheringShader(
    size,
    c0_size,
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
