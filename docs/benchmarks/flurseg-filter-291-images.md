# FlurSeg filter comparison on the 291-photo test project

The same exported 83,003,684 points were classified with FlurSeg before and after the
FlurSeg-specific post-filter changes. The existing SAM 3 classified LAS served as a reference.
SAM 3 is another model's output, not manually verified ground truth.

For comparison, SAM `road` and `sidewalk` map to FlurSeg `paved_surface`; SAM construction
machinery, material and scaffolding map to `construction`; SAM fence, pole, sign, power line
and water map to `other`. FlurSeg markings and utility covers have no matching SAM class, so
this score understates agreement on those features.

| Measure | Previous FlurSeg | Updated FlurSeg |
|---|---:|---:|
| Agreement with mapped SAM 3 labels | 82.8% | **87.5%** |
| SAM ground labelled ground | 24.5% | **56.0%** |
| SAM building labelled building | 70.6% | **87.5%** |
| SAM tree labelled tree | 44.1% | 45.2% |
| SAM low vegetation labelled low vegetation | 99.3% | 97.9% |
| SAM other labelled other | 64.8% | 54.0% |

The updated rules changed 850,971 of the 22,116,954 voxel voting points from low vegetation
to ground and 408,471 from other to building before the remaining geometry filters. The
ground correction uses the original point colour only for points Metashape classified as
terrain. The building correction requires a recognised building point within 0.75 m in 3D.
The colour threshold and building radius are in `config/profile.flurseg.yml`.

These rules were chosen using a 1% point sample and checked on separate 20 m spatial cells.
The complete LAS comparison above includes all points. Dry or discoloured vegetation in
another season can need a different colour threshold. Tree and kerb agreement remains low;
the post-filters cannot recover all detail missing from the image model's predictions.

The isolated voting-only reruns took 440.5 s before and 428.7 s after. They reused all 291
segmentation caches and do not measure fresh FlurSeg inference, export or Metashape import.
The original complete workflow comparison remains in the README.

## Per-class geometry review and export check

Measured on the same project with an isolated voting-only rerun (291 cached FlurSeg images,
83,003,684 points, 22,116,954 voting points). The SAM 3 reference survives only as every
100th source point (830,037 points), mapped as above. Labels were compared per source point
after propagation, so the full voting density was used, not the 1% sample alone.

| Measure | Without geometry review | With geometry review |
|---|---:|---:|
| Agreement with mapped SAM 3 labels | 87.50% | **88.17%** |
| SAM building labelled building (recall) | 87.5% | **92.5%** |
| FlurSeg building that SAM calls building (precision) | 92.7% | **92.8%** |

Changes of the geometry review (voting points), with the share of changed sample points that
SAM 3 labels as the new class:

| Change | Voting points | Agrees with SAM 3 |
|---|---:|---:|
| construction on roofs → building | 100,931 | 94% |
| other on facades → building | 48,152 | 90% |
| unclassified → building (facade 96%, roof 75%) | 5,256 | 92% |

Rejected after measurement: `other` on horizontal planes (14% building), `construction` on
facades (77%, often scaffolding), unclassified points without surrounding support (55% on roofs
before the 7-of-8-directions rule), and `low_vegetation` → building. A geometric fill of
unclassified ground or pavement was not added: 85% of the would-be pavement points are ground in
SAM 3, and geometry cannot separate two classes that share one plane. No changed point is a
SAM 3 kerb. The 25 cm clearance around kerbs, markings and covers keeps 10,700 more voting
points as `other` in the 0.75 m building-neighbour rule. Only 59% of them are SAM 3 building,
compared with 92% for the rule as a whole. The clearance costs 0.03 points of agreement.

Metashape's internal frame is tilted by 31.7 degrees from the vertical in this project. The
roof/facade split uses the true vertical, estimated from the point cloud (walls and planar
surfaces): 0.5–0.6 degrees from the CRS vertical, with the same agreement (88.215% against
88.214% with the CRS vertical). `postprocess.level_to_vertical: true` levels all
post-filters. It lowered agreement to 88.05%, mainly through tree/low-vegetation swaps.
The SAM 3 reference was produced by the same tilted DTM filter, so this comparison cannot judge
the levelled frame; the option stays off until the thresholds are recalibrated.

| Stage 2 (voting-only) | Before | After |
|---|---:|---:|
| Total | 418.4 s | 415.4 s |
| Post-filter | 71.6 s | **53.6 s** |
| Geometry review within it | 4.2 s | 3.7 s |
| Peak working set | 7.04 GB | **6.67 GB** |

The post-filter gained from a closed-form plane fit in the DTM and ground filters: labels are
identical on all 22,116,954 voting points. Normals are kept in float32, and the ground hint
comes from the same LAS scan as the other attributes. Voting time varies by ±10% between runs.

The delivered `export/punktwolke_class.las` had the correct codes but carried the key-point
flag on all 12,688,166 points with codes of 32 or more, a side effect of Metashape's LAS 1.2
class parts. Reassembled from the same parts, the file has identical codes and no such flag.
Writing it took 21 s; the format and class check took 2 ms. `export/Flurseg.laz` (27,167,472
points, LAS 1.2 point format 2, written by Metashape) cannot hold codes above 31.
