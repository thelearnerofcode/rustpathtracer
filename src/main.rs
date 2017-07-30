extern crate nalgebra;
extern crate image;
extern crate rand;
extern crate cpuprofiler;
extern crate crossbeam;
extern crate imageproc;

use std::path::Path;
use std::fs::File;
use std::ops::{Add, AddAssign, Div, Mul};
use std::{f32, f64};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nalgebra::Vector3;
use image::{DynamicImage, GenericImage, Pixel, Rgba};
use rand::distributions::{IndependentSample, Range};

const GAMMA: f32 = 2.2;

fn gamma_encode(linear: f32) -> f32 {
    linear.powf(1.0 / GAMMA)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub red: f32,
    pub green: f32,
    pub blue: f32,
}

impl Color {
    pub fn new(red: f32, green: f32, blue: f32) -> Color {
        Color {
            red: red,
            green: green,
            blue: blue,
        }
    }
    pub fn black() -> Color {
        Color {
            red: 0.0,
            green: 0.0,
            blue: 0.0,
        }
    }
    pub fn white() -> Color {
        Color {
            red: 1.0,
            green: 1.0,
            blue: 1.0,
        }
    }
    pub fn to_rgba(&self) -> Rgba<u8> {
        Rgba::from_channels(
            (gamma_encode(self.red) * 255.0) as u8,
            (gamma_encode(self.green) * 255.0) as u8,
            (gamma_encode(self.blue) * 255.0) as u8,
            255,
        )
    }
}

impl Add for Color {
    type Output = Color;
    fn add(self, other: Color) -> Color {
        Color {
            red: self.red + other.red,
            green: self.green + other.green,
            blue: self.blue + other.blue,
        }
    }
}

impl AddAssign for Color {
    fn add_assign(&mut self, other: Color) {
        *self = Color {
            red: self.red + other.red,
            green: self.green + other.green,
            blue: self.blue + other.blue,
        }
    }
}

impl Mul for Color {
    type Output = Color;
    fn mul(self, other: Color) -> Color {
        Color {
            red: self.red * other.red,
            green: self.green * other.green,
            blue: self.blue * other.blue,
        }
    }
}

impl Mul<f32> for Color {
    type Output = Color;
    fn mul(self, other: f32) -> Color {
        Color {
            red: self.red * other,
            green: self.green * other,
            blue: self.blue * other,
        }
    }
}

impl Mul<f64> for Color {
    type Output = Color;
    fn mul(self, other: f64) -> Color {
        Color {
            red: self.red * other as f32,
            green: self.green * other as f32,
            blue: self.blue * other as f32,
        }
    }
}

impl Mul<Color> for f64 {
    type Output = Color;
    fn mul(self, other: Color) -> Color {
        Color {
            red: other.red * self as f32,
            green: other.green * self as f32,
            blue: other.blue * self as f32,
        }
    }
}

impl Div<f32> for Color {
    type Output = Color;

    fn div(self, other: f32) -> Color {
        Color {
            red: self.red / other,
            green: self.green / other,
            blue: self.blue / other,
        }
    }
}

impl Div<f64> for Color {
    type Output = Color;

    fn div(self, other: f64) -> Color {
        Color {
            red: self.red / other as f32,
            green: self.green / other as f32,
            blue: self.blue / other as f32,
        }
    }
}
impl Div<i32> for Color {
    type Output = Color;

    fn div(self, other: i32) -> Color {
        Color {
            red: self.red / other as f32,
            green: self.green / other as f32,
            blue: self.blue / other as f32,
        }
    }
}

#[derive(Debug)]
pub enum MaterialType {
    Specular,
    Diffuse,
    Metal,
    Coat,
}

#[derive(Debug)]
pub struct Material {
    pub color: Color,
    pub emission: Color,
    pub material_type: MaterialType,
}

#[derive(Debug)]
pub struct Sphere {
    pub center: Vector3<f64>,
    pub radius: f64,
    pub material: Material,
    pub edge_size: f64,
}

impl Sphere {
    fn surface_normal(&self, hit_point: &Vector3<f64>) -> Vector3<f64> {
        (*hit_point - self.center).normalize()
    }
}

#[derive(Debug)]
pub struct Plane {
    pub origin: Vector3<f64>,
    pub normal: Vector3<f64>,
    pub material: Material,
}

impl Plane {
    fn surface_normal(&self, _: &Vector3<f64>) -> Vector3<f64> {
        -self.normal.normalize()
    }
}

#[derive(Debug)]
pub struct Ray {
    pub origin: Vector3<f64>,
    pub direction: Vector3<f64>,
}

impl Ray {
    pub fn create_prime(x: u32, y: u32, scene: &Scene) -> Ray {
        let fov_adjustment = (scene.fov.to_radians() / 2.0).tan();
        let aspect_ratio = (scene.width as f64) / (scene.height as f64);
        let sensor_x =
            ((((x as f64 + 0.5) / scene.width as f64) * 2.0 - 1.0) * aspect_ratio) * fov_adjustment;
        let sensor_y = (1.0 - ((y as f64 + 0.5) / scene.height as f64) * 2.0) * fov_adjustment;

        Ray {
            origin: Vector3::new(0.0, 0.0, 0.0),
            direction: Vector3::new(sensor_x, sensor_y, -1.0).normalize(),
        }
    }
}

pub trait Intersectable {
    fn intersect(&self, ray: &Ray) -> Option<f64>;
    fn intersect_edge(&self, ray: &Ray) -> bool;
}

impl Intersectable for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        let l: Vector3<f64> = self.center - ray.origin;
        let adj = l.dot(&ray.direction);
        let d2 = l.dot(&l) - (adj * adj);
        let radius2 = self.radius * self.radius;
        if d2 > radius2 {
            return None;
        }
        let thc = (radius2 - d2).sqrt();
        let t0 = adj - thc;
        let t1 = adj + thc;

        if t0 < 0.0 && t1 < 0.0 {
            return None;
        }

        let distance = if t0 < t1 { t0 } else { t1 };
        Some(distance)
    }
    fn intersect_edge(&self, ray: &Ray) -> bool {
        if self.edge_size != 0.0 {
            let smaller_sphere = Sphere {
                radius: self.radius - self.edge_size,
                center: self.center,
                material: Material {
                    color: Color::white(),
                    emission: Color::black(),
                    material_type: MaterialType::Diffuse,
                },
                edge_size: 0.0,
            };
            let test_intersection = smaller_sphere.intersect(ray);
            match test_intersection {
                Some(_) => false,
                None => true,
            }
        } else {
            false
        }
    }
}

#[derive(Debug)]
pub enum Object {
    Sphere(Sphere),
    Plane(Plane),
}

impl Object {
    pub fn material(&self) -> &Material {
        match *self {
            Object::Sphere(ref s) => &s.material,
            Object::Plane(ref p) => &p.material,
        }
    }

    pub fn surface_normal(&self, hit_point: &Vector3<f64>) -> Vector3<f64> {
        let surface_normal: Vector3<f64>;
        match *self {
            Object::Sphere(ref s) => surface_normal = s.surface_normal(hit_point),
            Object::Plane(ref p) => surface_normal = p.surface_normal(hit_point),
        }
        surface_normal
    }
}

pub struct Intersection<'a> {
    pub distance: f64,
    pub hit_point: Vector3<f64>,
    pub object: &'a Object,
}

impl<'a> Intersection<'a> {
    pub fn new<'b>(distance: f64, object: &'b Object, ray: &Ray) -> Intersection<'b> {
        Intersection {
            distance: distance,
            object: object,
            hit_point: ray.origin + (ray.direction * distance),
        }
    }
}


impl Intersectable for Object {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        match *self {
            Object::Sphere(ref s) => s.intersect(ray),
            Object::Plane(ref p) => p.intersect(ray),
        }
    }
    fn intersect_edge(&self, ray: &Ray) -> bool {
        match *self {
            Object::Sphere(ref s) => s.intersect_edge(ray),
            Object::Plane(ref p) => p.intersect_edge(ray),
        }
    }
}

impl Intersectable for Plane {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        let normal = &self.normal;
        let denom = normal.dot(&ray.direction);
        if denom > 1e-6 {
            let v = self.origin - ray.origin;
            let distance = v.dot(&normal) / denom;
            if distance >= 0.0 {
                return Some(distance);
            }
        }
        None
    }

    fn intersect_edge(&self, _: &Ray) -> bool {
        false
    }
}

trait LightTrait {
    // illuminate takes the point to illuminate and returns the lights
    // direction, intensity, and distance
    fn illuminate(&self, Vector3<f64>) -> (Vector3<f64>, Color, f64);
}

#[derive(Debug)]
pub struct DistantLight {
    pub direction: Vector3<f64>,
    pub color: Color,
    pub intensity: f32,
}

impl LightTrait for DistantLight {
    fn illuminate(&self, _: Vector3<f64>) -> (Vector3<f64>, Color, f64) {
        let light_dir = self.direction;
        let intensity = self.color * self.intensity;
        let distance = 1.0f64 / 0.0f64;
        (light_dir, intensity, distance)
    }
}

#[derive(Debug)]
pub struct PointLight {
    pub position: Vector3<f64>,
    pub color: Color,
    pub intensity: f32,
}

impl LightTrait for PointLight {
    fn illuminate(&self, point: Vector3<f64>) -> (Vector3<f64>, Color, f64) {
        let light_dir = point - self.position;
        let r2 = light_dir.norm();
        let distance = r2.sqrt();
        let intensity = self.color * self.intensity / (4.0 * f64::consts::PI * r2);
        (light_dir, intensity, distance)
    }
}

#[derive(Debug)]
pub enum Light {
    DistantLight(DistantLight),
    PointLight(PointLight),
}
impl LightTrait for Light {
    fn illuminate(&self, point: Vector3<f64>) -> (Vector3<f64>, Color, f64) {
        match self {
            &Light::DistantLight(ref light) => light.illuminate(point),
            &Light::PointLight(ref light) => light.illuminate(point),
        }
    }
}

// Contains scene data
pub struct Scene {
    pub width: u32,
    pub height: u32,
    pub fov: f64,
    pub objects: Vec<Object>,
    pub lights: Vec<Light>,
    pub samples: i32,
    pub bias: f64,
}

fn orient_normal(normal: &Vector3<f64>, ray: &Ray) -> Vector3<f64> {
    if normal.dot(&ray.direction) < 0.0 {
        *normal
    } else {
        -normal
    }
}

fn create_coordinate_system(normal: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let u: Vector3<f64>;

    if normal.x.abs() > 0.1 {
        u = Vector3::new(0.0, 1.0, 0.0).cross(&normal);
    } else {
        u = Vector3::new(1.0, 0.0, 0.0).cross(&normal);
    }

    let u = u.normalize();
    let v = normal.cross(&u);
    (u, v)
}

impl Scene {
    pub fn trace(&self, ray: &Ray) -> Option<Intersection> {
        self.objects
            .iter()
            .filter_map(|s| s.intersect(ray).map(|d| Intersection::new(d, s, ray)))
            .min_by(|i1, i2| i1.distance.partial_cmp(&i2.distance).unwrap())
    }

    pub fn cast_ray(&self, ray: &Ray, depth: u64) -> Color {
        let intersection_option = self.trace(ray);
        match intersection_option {
            Some(intersection) => {
                // gen rng
                let between = Range::new(0.0f64, 1.0f64);
                let mut rng = rand::thread_rng();

                // find material and color of intersected object
                let material = &intersection.object.material();
                let objcolor = material.color;
                let emission = material.emission;

                // calculate max reflection
                let p: f32;
                if objcolor.red > objcolor.green && objcolor.red > objcolor.blue {
                    p = objcolor.red;
                } else {
                    if objcolor.green > objcolor.blue {
                        p = objcolor.green;
                    } else {
                        p = objcolor.blue;
                    }
                }
                let mut color: Color;
                if depth > 5 {
                    if (between.ind_sample(&mut rng) as f32) < p {
                        color = objcolor * (1.0 / p)
                    } else {
                        return emission;
                    }
                } else {
                    color = objcolor;
                };
                // check if intersected with edge
                if intersection.object.intersect_edge(&ray) {
                    color = color * 0.05;
                }

                // calculate normal
                let normal = intersection.object.surface_normal(&intersection.hit_point);
                let oriented_normal = orient_normal(&normal, &ray);

                match material.material_type {
                    MaterialType::Diffuse => {
                        let r1 = 2.0 * f64::consts::PI * between.ind_sample(&mut rng);
                        let r2 = between.ind_sample(&mut rng);
                        let r2s = r2.sqrt();

                        let (u, v) = create_coordinate_system(oriented_normal.normalize());

                        let direction = ((u * r1.cos() * r2s) + (v * r1.sin() * r2s) +
                            (oriented_normal * (1.0 - r2).sqrt()))
                            .normalize();

                        let reflected_ray = Ray {
                            direction: direction,
                            origin: intersection.hit_point + (oriented_normal * self.bias),
                        };
                        let reflected_color = self.cast_ray(&reflected_ray, depth + 1);

                        emission + color * reflected_color
                    }
                    MaterialType::Specular => {
                        let reflected_ray = Ray {
                            direction: ray.direction - normal * 2.0 * normal.dot(&ray.direction),
                            origin: intersection.hit_point + (oriented_normal * self.bias),
                        };
                        let reflected_color = self.cast_ray(&reflected_ray, depth + 1);
                        emission + color * reflected_color
                    }
                    MaterialType::Metal => {
                        let phi = 2.0 * f64::consts::PI * between.ind_sample(&mut rng);
                        let r2 = between.ind_sample(&mut rng);

                        let phongexponent = 30.0;
                        let cos_theta = (1.0 - r2).powf(1.0 / (phongexponent + 1.0));
                        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

                        let w = ray.direction - normal * 2.0 * normal.dot(&ray.direction);
                        let w = w.normalize();

                        let (u, v) = create_coordinate_system(w);

                        let reflected_ray = Ray {
                            direction: u * phi.cos() * sin_theta + v * phi.sin() * sin_theta +
                                oriented_normal * cos_theta,
                            origin: intersection.hit_point + (oriented_normal * self.bias),
                        };

                        let reflected_color = self.cast_ray(&reflected_ray, depth + 1);
                        emission + color * reflected_color
                    }
                    MaterialType::Coat => {
                        let random_float = between.ind_sample(&mut rng);
                        let threshold = 0.05;
                        let reflect_from_surface = random_float < threshold;
                        if reflect_from_surface {
                            let reflected_ray = Ray {
                                direction: ray.direction -
                                    normal * 2.0 * normal.dot(&ray.direction),
                                origin: intersection.hit_point + (oriented_normal * self.bias),
                            };
                            let reflected_color = self.cast_ray(&reflected_ray, depth + 1);
                            emission + color * reflected_color
                        } else {
                            let r1 = 2.0 * f64::consts::PI * between.ind_sample(&mut rng);
                            let r2 = between.ind_sample(&mut rng);
                            let r2s = r2.sqrt();

                            let (u, v) = create_coordinate_system(oriented_normal.normalize());

                            let direction = ((u * r1.cos() * r2s) + (v * r1.sin() * r2s) +
                                (oriented_normal * (1.0 - r2).sqrt()))
                                .normalize();

                            let reflected_ray = Ray {
                                direction: direction,
                                origin: intersection.hit_point + (oriented_normal * self.bias),
                            };
                            let reflected_color = self.cast_ray(&reflected_ray, depth + 1);

                            emission + color * reflected_color
                        }
                    }
                }
            }
            None => Color::white(),
        }
    }

    fn render_pixel(&self, x: u32, y: u32) -> Color {
        let mut final_color = Color::black();
        for _ in 0..self.samples {
            let ray = Ray::create_prime(x, y, self);
            final_color = final_color + self.cast_ray(&ray, 0);
        }
        final_color / self.samples
    }

    pub fn render(&self) -> DynamicImage {
        // create progress bar
        let total_pixels = self.width * self.height;
        // pixel_data stores the position and color of each pixel
        let pixel_data: Arc<Mutex<Vec<(u32, u32, Color)>>> =
            Arc::new(Mutex::new(Vec::with_capacity(total_pixels as usize)));

        crossbeam::scope(|scope| for x in 0..self.width {
            let pixel_data = Arc::clone(&pixel_data);
            scope.spawn(move || {
                let mut row: Vec<(u32, u32, Color)> = Vec::with_capacity(self.height as usize);
                for y in 0..self.height {
                    // render pixel
                    let color = self.render_pixel(x, y);
                    row.push((x, y, color));
                }
                pixel_data.lock().unwrap().extend(row.as_slice());
                // increase progress bar
            });
        });
        let mut image = DynamicImage::new_rgb8(self.width, self.height);
        let pixel_data = pixel_data.lock().unwrap();
        for &(x, y, color) in pixel_data.iter() {
            image.put_pixel(x, y, color.to_rgba());
        }
        image
    }
}

fn main() {
    let scene = Scene {
        width: 1920,
        height: 1080,
        fov: 45.0,
        objects: vec![
            // black sphere
            Object::Sphere(Sphere {
                center: Vector3::new(-1.5, 0.0, -5.0),
                radius: 1.0,
                material: Material {
                    color: Color::white() * 0.5,
                    emission: Color::black(),
                    material_type: MaterialType::Diffuse,
                },
                edge_size: 0.015,
            }),
            Object::Sphere(
                // blue sphere
                Sphere {
                    center: Vector3::new(1.5, 0.0, -5.0),
                    radius: 1.0,
                    material: Material {
                        color: Color::new(0.529411764706, 0.807843137255, 0.980392156863),
                        emission: Color::black(),
                        material_type: MaterialType::Specular,
                    },
                    edge_size: 0.015,
                },
            ),
            Object::Plane(
                // background
                Plane {
                    origin: Vector3::new(1.0, -3.1, 0.0),
                    normal: Vector3::new(0.0, -110.0, -50.0),
                    material: Material {
                        color: Color::white() * 0.1,
                        emission: Color::black(),
                        material_type: MaterialType::Coat,
                    },
                },
            ),
            Object::Sphere(
                // light
                Sphere {
                    center: Vector3::new(0.0, 10.0, -10.0),
                    radius: 1.0,
                    material: Material {
                        color: Color::white() * 0.2,
                        emission: Color::new(1.0, 1.0, 1.0),
                        material_type: MaterialType::Diffuse,
                    },
                    edge_size: 0.0,
                },
            ),
        ],
        lights: vec![
            Light::DistantLight(DistantLight {
                color: Color::white(),
                direction: Vector3::new(1.5, -5.0, -5.0).normalize(),
                intensity: 1.0,
            }),
        ],
        samples: 512,
        bias: 0.0001,
    };
    println!("Starting Renderer.");
    let now = Instant::now();
    let img: DynamicImage = scene.render();
    println!("Completed in {} seconds.", now.elapsed().as_secs());

    let file_name = "render_test1.png";
    let ref mut fout = File::create(&Path::new(&file_name)).unwrap();
    // Write the contents of this image to the Writer in PNG format.
    let _ = img.save(fout, image::PNG).unwrap();
}
