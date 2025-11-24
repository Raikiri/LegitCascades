// Available at
// https://radiance-cascades.github.io/LegitScriptEditor/?gh=Raikiri/LegitCascades/EnergyConservingHRC.ls

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

[include: "probe_hrc_grid"]
[declaration: "energy_conserving_hrc"]
{{
  struct PixelInterval
  {
    vec2 start_pixel_pos;
    vec2 end_pixel_pos;
  };
  vec2 GetProbePixelPos(float probe_idxf, float plane_idxf, CascadeGeom cascade_geom)
  {
    return vec2(
      plane_idxf * float(cascade_geom.plane_spacing),
      (probe_idxf + 0.5f) * float(cascade_geom.probe_spacing));
  }

  struct ProbeLocf
  {
    float probe_idxf;
    float plane_idxf;
  };
  ProbeLocf GetProbeLocf(vec2 pixel_pos, CascadeGeom cascade_geom)
  {
    ProbeLocf probe_locf;
    probe_locf.plane_idxf = pixel_pos.x / float(cascade_geom.plane_spacing);
    probe_locf.probe_idxf = pixel_pos.y / float(cascade_geom.probe_spacing) - 0.5f;
    return probe_locf;
  }

  float GetProbeDirIdxf(vec2 pixel_delta, uint dirs_count)
  {
    float ratio = pixel_delta.y / pixel_delta.x * 0.5f + 0.5f;
    return ratio * float(dirs_count) - 0.5f;
  }
  float GetProbeDirIdxf(float probe_idxf, float plane_idxf, vec2 pixel_pos, CascadeGeom cascade_geom, CascadeSize cascade_size)
  {
    vec2 probe_pixel_pos = GetProbePixelPos(probe_idxf, plane_idxf, cascade_geom);
    return GetProbeDirIdxf(pixel_pos - probe_pixel_pos, cascade_geom.dirs_count);
  }
  vec4 TraceInterval(IntervalIdx interval_idx, vec2 light_pixel_pos, float light_pixel_height, CascadeGeom cascade_geom, CascadeSize cascade_size)
  {
    vec2 probe_pixel_pos = GetProbePixelPos(float(interval_idx.probe_idx), float(interval_idx.plane_idx), cascade_geom);
    vec2 next_pixel_pos = GetProbePixelPos(float(interval_idx.probe_idx), float(interval_idx.plane_idx + 1), cascade_geom);
    
    vec2 light_start_pixel_pos = light_pixel_pos + vec2(0.0f, -light_pixel_height / 2.0f);
    vec2 light_end_pixel_pos = light_pixel_pos + vec2(0.0f,  light_pixel_height / 2.0f);

    if(light_pixel_pos.x < probe_pixel_pos.x || light_pixel_pos.x > next_pixel_pos.x)
    {
      return vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    float start_dir_idxf = GetProbeDirIdxf(light_start_pixel_pos - probe_pixel_pos, cascade_geom.dirs_count);
    float end_dir_idxf = GetProbeDirIdxf(light_end_pixel_pos - probe_pixel_pos, cascade_geom.dirs_count);

    float occlusion = max(0.0f, 
      min(end_dir_idxf,    (float(interval_idx.dir_idx) + 0.5f)) -
      max(start_dir_idxf,  (float(interval_idx.dir_idx) - 0.5f)));
    
    return vec4(vec3(1.0f, 0.5f, 0.0f) * occlusion, 1.0f - occlusion);
  }
}}

[include: "pcg", "energy_conserving_hrc", "merging"]
void MergeCascade(
  uvec2 viewport_size,
  uint dst_cascade_idx,
  uint do_merge,
  vec2 light_pixel_pos,
  float light_pixel_height,
  out vec4 dst_merged_interval,
  sampler2D src_merged_img)
{{
  CascadeGeom c0_geom = GetC0CascadeGeom();
  CascadeSize c0_size = GetC0CascadeSize(viewport_size, c0_geom);

  CascadeGeom dst_cascade_geom = GetCascadeGeom(c0_geom, dst_cascade_idx);
  CascadeSize dst_cascade_size = GetCascadeSize(c0_size, dst_cascade_idx);

  uint src_cascade_idx = dst_cascade_idx + 1u;
  CascadeGeom src_cascade_geom = GetCascadeGeom(c0_geom, src_cascade_idx);
  CascadeSize src_cascade_size = GetCascadeSize(c0_size, src_cascade_idx);
  
  ivec2 dst_texel_idx = ivec2(gl_FragCoord.xy);
  IntervalIdx dst_interval_idx = AtlasTexelIdxToIntervalIdx(dst_texel_idx, dst_cascade_size);
  if(
    dst_interval_idx.probe_idx < int(dst_cascade_size.probes_count) &&
    dst_interval_idx.plane_idx < int(dst_cascade_size.planes_count) &&
    dst_interval_idx.dir_idx < int(dst_cascade_size.dirs_count))
  {
    vec4 src_merged_interval = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    if(do_merge == 1u)
    {
      if(dst_interval_idx.plane_idx % 2 == 0)
      {
        IntervalIdx src_interval_idx;
        src_interval_idx.probe_idx = dst_interval_idx.probe_idx;
        src_interval_idx.plane_idx = dst_interval_idx.plane_idx / 2;
        for(uint dir_num; dir_num < 2u; dir_num++)
        {
          src_interval_idx.dir_idx = dst_interval_idx.dir_idx * 2 + int(dir_num);
          src_merged_interval += ReadInterval(src_merged_img, src_interval_idx, src_cascade_size) * 0.5f;
        }
      }else
      {
        IntervalIdx src_interval_idx;
        src_interval_idx.plane_idx = (dst_interval_idx.plane_idx + 1) / 2;
        for(uint dir_num; dir_num < 2u; dir_num++)
        {
          src_interval_idx.dir_idx = dst_interval_idx.dir_idx * 2 + int(dir_num);
          src_interval_idx.probe_idx = dst_interval_idx.probe_idx - int(dst_cascade_geom.dirs_count / 2u) + dst_interval_idx.dir_idx;
          src_merged_interval += ReadInterval(src_merged_img, src_interval_idx, src_cascade_size) * 0.5f;
        }
        /*for(uint dir_num; dir_num < 2u; dir_num++)
        {
          src_interval_idx.dir_idx = dst_interval_idx.dir_idx * 2 + int(dir_num);
          src_interval_idx.probe_idx = dst_interval_idx.probe_idx - int(dst_cascade_geom.dirs_count / 2u) + dst_interval_idx.dir_idx + 1;
          src_merged_interval += ReadInterval(src_merged_img, src_interval_idx, src_cascade_size) * 0.25f;
        }*/
      }
    }
    //int src_c0_plane_idx = (interval_idx.plane_idx + 1) * int(c0_geom.plane_spacing);
    //int src_c0_probe_idx = interval_idx 
    vec4 dst_interval = TraceInterval(dst_interval_idx, light_pixel_pos, light_pixel_height, dst_cascade_geom, dst_cascade_size);
    dst_merged_interval = MergeIntervals(dst_interval, src_merged_interval);
  }else
  {
    dst_merged_interval = vec4(0.1f, 0.0f, 0.0f, 1.0f);
  }
}}

[include: "pcg", "energy_conserving_hrc"]
void FinalGatheringShader(
  uvec2 viewport_size,
  vec2 light_pixel_pos,
  float light_pixel_height,
  sampler2D merged_atlas_img,
  uint cascade_idx,
  out vec4 color)
{{
  //vec2 screen_pos = gl_FragCoord.xy;
  //uint cascade_idx = 0u;

  CascadeGeom c0_geom = GetC0CascadeGeom();
  CascadeSize c0_size = GetC0CascadeSize(viewport_size, c0_geom);

  CascadeGeom cascade_geom = GetCascadeGeom(c0_geom, cascade_idx);
  CascadeSize cascade_size = GetCascadeSize(c0_size, cascade_idx);

  ProbeLocf probe_locf = GetProbeLocf(gl_FragCoord.xy, cascade_geom);

  vec4 fluence = vec4(0.0f);
  for(uint dir_idx = 0u; dir_idx < cascade_size.dirs_count; dir_idx++)
  {
    IntervalIdx interval_idx;
    interval_idx.probe_idx = int(floor(probe_locf.probe_idxf + 0.5f));
    interval_idx.plane_idx = int(floor(probe_locf.plane_idxf)) + 1;
    interval_idx.dir_idx = int(dir_idx);
    fluence += ReadInterval(merged_atlas_img, interval_idx, cascade_size) / float(cascade_size.dirs_count);
  }
  color = fluence;

  vec2 light_delta = light_pixel_pos - gl_FragCoord.xy;
  if(abs(light_delta.x) < 2.0f && abs(light_delta.y) < light_pixel_height / 2.0f + 2.0f)
  {
    color = vec4(1.0f, 0.5f, 0.0f, 1.0f);
  }
}}

[rendergraph]
[include: "fps"]
void RenderGraphMain()
{{
  uvec2 viewport_size = GetSwapchainImage().GetSize();
  ClearShader(GetSwapchainImage());
  uvec2 c0_atlas_size = viewport_size;
  //c0_size.x = SliderInt("c0_size.x", 1, 1024, 256);
  //c0_size.y = size.y * c0_size.x / size.x;
  vec2 light_pixel_pos;
  light_pixel_pos.x = SliderFloat("light_pixel_pos.x", 0.0f, 512.0f, 100.0f);
  light_pixel_pos.y = SliderFloat("light_pixel_pos.y", 0.0f, 512.0f, 100.0f);
  float light_pixel_height = SliderFloat("light_pixel_height", 0.0f, 512.0f, 100.0f);

  const int cascades_count = 6;

  Image src_merged_img = GetImage(c0_atlas_size, rgba16f);
  Image dst_merged_img = GetImage(c0_atlas_size, rgba16f);

  for(uint cascade_num = 0; cascade_num < cascades_count; cascade_num++)
  {
    uint dst_cascade_idx = cascades_count - 1 - cascade_num;
    bool do_merge = cascade_num != 0;
    MergeCascade(viewport_size, dst_cascade_idx, do_merge ? 1 : 0, light_pixel_pos, light_pixel_height, dst_merged_img, src_merged_img);
    CopyShader(dst_merged_img, src_merged_img);
  }
  //SceneShader(size, scene_img);

  // uvec2 atlas_size = GetAtlasSize(cascade_scaling_pow2, cascades_count, c0_size);
  // Text("c0_size: " + c0_size + " atlas size: " + atlas_size);
  // Image merged_atlas_img = GetImage(atlas_size, rgba16f);
  // Image prev_atlas_img = GetImage(atlas_size, rgba16f);

  FinalGatheringShader(
    viewport_size,
    light_pixel_pos,
    light_pixel_height,
    src_merged_img,
    0,
    GetSwapchainImage()
  );
  /*OverlayTexShader(
    dst_merged_img,
    GetSwapchainImage());*/

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
[declaration: "probe_hrc_grid"]
{{
  struct CascadeSize
  {
    uint probes_count;
    uint planes_count;
    uint dirs_count;
  };

  struct CascadeGeom
  {
    uint probe_spacing;
    uint plane_spacing;
    uint dirs_count;
  };

  CascadeSize GetCascadeSize(CascadeSize c0_cascade_size, uint cascade_idx)
  {
    CascadeSize cascade_size;
    cascade_size.probes_count = c0_cascade_size.probes_count;
    cascade_size.planes_count = c0_cascade_size.planes_count >> cascade_idx;
    cascade_size.dirs_count = c0_cascade_size.dirs_count << cascade_idx;
    return cascade_size;
  }
  CascadeGeom GetCascadeGeom(CascadeGeom c0_cascade_geom, uint cascade_idx)
  {
    CascadeGeom cascade_geom;
    cascade_geom.probe_spacing = c0_cascade_geom.probe_spacing;
    cascade_geom.plane_spacing = c0_cascade_geom.plane_spacing << cascade_idx;
    cascade_geom.dirs_count = c0_cascade_geom.dirs_count << cascade_idx;
    return cascade_geom;
  }

  CascadeGeom GetC0CascadeGeom()
  {
    CascadeGeom cascade_geom;
    cascade_geom.probe_spacing = 8u;
    cascade_geom.plane_spacing = 8u;
    cascade_geom.dirs_count = 2u;
    return cascade_geom;
  }
  CascadeSize GetC0CascadeSize(uvec2 viewport_size, CascadeGeom c0_cascade_geom)
  {
    CascadeSize c0_cascade_size;
    c0_cascade_size.probes_count = viewport_size.y / c0_cascade_geom.probe_spacing;
    c0_cascade_size.planes_count = viewport_size.x / c0_cascade_geom.plane_spacing;
    c0_cascade_size.dirs_count = c0_cascade_geom.dirs_count;
    return c0_cascade_size;
  }
  uvec2 GetAtlasSize(CascadeSize cascade_size)
  {
    return uvec2(cascade_size.planes_count * cascade_size.dirs_count, cascade_size.probes_count);
  }

  struct IntervalIdx
  {
    int probe_idx;
    int plane_idx;
    int dir_idx;
  };

  ivec2 IntervalIdxToAtlasTexelIdx(IntervalIdx interval_idx, CascadeSize cascade_size)
  {
    return ivec2(interval_idx.plane_idx * int(cascade_size.dirs_count) + interval_idx.dir_idx, interval_idx.probe_idx);
  }

  IntervalIdx AtlasTexelIdxToIntervalIdx(ivec2 texel_idx, CascadeSize cascade_size)
  {
    IntervalIdx interval_idx;
    interval_idx.probe_idx = texel_idx.y;
    interval_idx.plane_idx = texel_idx.x / int(cascade_size.dirs_count);
    interval_idx.dir_idx = texel_idx.x % int(cascade_size.dirs_count);
    return interval_idx;
  }

  vec4 ReadInterval(sampler2D cascade_atlas_img, IntervalIdx interval_idx, CascadeSize cascade_size)
  {
    ivec2 texel_idx = IntervalIdxToAtlasTexelIdx(interval_idx, cascade_size);
    return texelFetch(cascade_atlas_img, texel_idx, 0);
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
